from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch

from pioneerml.integration.pytorch.modules.factory.registry import REGISTRY as MODULE_REGISTRY
from pioneerml.integration.pytorch.modules.graph_lightning_module import GraphLightningModule


@MODULE_REGISTRY.register("purity_multilevel")
@MODULE_REGISTRY.register("purity_graph_lightning")
class PurityMultiLevelLightningModule(GraphLightningModule):
    """PURITY Lightning module that aligns loss on multi-level outputs directly."""

    def __init__(
        self,
        *args,
        optimizer_cls: type[optim.Optimizer] | None = None,
        optimizer_param_groups: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        # Keep Dropout layers stochastic during validation (MC-dropout). Graph nets
        # are trained with dropout on, so evaluating with it off measures a different,
        # lower-variance network than the one being optimized; keeping it on makes the
        # val loss share the train dropout regime. Scoped to the Lightning val metric
        # only — the evaluator/export call model.eval() directly and stay deterministic.
        # Default on; set mc_dropout_in_val=False to restore eval-mode validation.
        mc_dropout_in_val = kwargs.pop("mc_dropout_in_val", True)
        # V2 parity: both reference trainers (train_purity.py / train_fast3_staged.py)
        # build the optimizer with torch.optim.AdamW (decoupled weight decay). The earlier
        # "Omar parity: Adam" override was incorrect — neither reference uses Adam.
        if optimizer_cls is None:
            optimizer_cls = optim.AdamW
        super().__init__(*args, optimizer_cls=optimizer_cls, **kwargs)
        self._mc_dropout_in_val = bool(mc_dropout_in_val)
        self._last_token_batch: torch.Tensor | None = None
        self._last_token_valid: torch.Tensor | None = None
        self._task_weights: dict[str, float] | None = None
        self.optimizer_param_groups: list[dict[str, Any]] = [
            dict(item) for item in list(optimizer_param_groups or [])
        ]
        self.pdg_class_names: tuple[str, ...] = ("pion", "muon", "mip")
        self.train_pdg_batch_accuracy_history: list[dict[str, Any]] = []
        self.val_pdg_batch_accuracy_history: list[dict[str, Any]] = []
        self.train_endpoint_mse_batch_history: list[dict[str, Any]] = []
        self.val_endpoint_mse_batch_history: list[dict[str, Any]] = []
        self.train_task_diagnostics_batch_history: list[dict[str, Any]] = []
        self.val_task_diagnostics_batch_history: list[dict[str, Any]] = []

    def set_task_weights(self, task_weights: Mapping[str, float] | None) -> None:
        if task_weights is None:
            self._task_weights = None
        else:
            out: dict[str, float] = {}
            for key, value in dict(task_weights).items():
                out[str(key)] = float(value)
            self._task_weights = out

        loss_setter = getattr(self.loss_fn, "set_task_weights", None)
        if callable(loss_setter):
            loss_setter(self._task_weights)

    def _enable_val_dropout(self) -> None:
        """Keep EVERY dropout source stochastic during validation (MC-dropout) after
        Lightning has put the model in eval mode. We deliberately do NOT flip the
        module's global `self.training` flag: that also gates teacher forcing and the
        inference path in the model adapter, so flipping it would make validation
        artificially easy. Instead we set train mode on exactly the dropout-bearing
        modules, whose ONLY train/eval-dependent behaviour is dropout.

        A full layer census of the model confirms three dropout sources (and no
        BatchNorm, so nothing else is affected):
          * nn.Dropout (FFN / head / transformer-layer dropout)        -> _DropoutNd
          * nn.MultiheadAttention (cross-attn + transformer self-attn) -> functional
          * torch_geometric TransformerConv (GNN attention dropout)    -> functional
        If a new dropout-bearing layer type is ever added, extend this set.
        """
        from torch.nn.modules.dropout import _DropoutNd

        try:
            from torch_geometric.nn import TransformerConv
        except Exception:  # pragma: no cover - torch_geometric is always present here
            TransformerConv = ()

        dropout_types: tuple[type, ...] = (_DropoutNd, torch.nn.MultiheadAttention)
        if TransformerConv:
            dropout_types = dropout_types + (TransformerConv,)

        n_active = 0
        for module in self.model.modules():
            if isinstance(module, dropout_types):
                module.train(True)
                n_active += 1
        if not getattr(self, "_logged_val_dropout", False):
            self._logged_val_dropout = True
            print(
                f"[purity] MC-dropout in validation: {n_active} dropout-bearing modules "
                "kept active (teacher forcing stays off)",
                flush=True,
            )

    def _model_forward(self, batch: Batch) -> Any:
        model = self.model
        task_weights = self._task_weights
        if task_weights:
            try:
                return model(batch, task_weights=task_weights)
            except TypeError:
                return model(batch)
        return model(batch)

    def compute_loss(self, raw_preds: Any, batch: Batch) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        try:
            out = self.loss_fn(raw_preds, batch)
            return self._normalize_loss_output(out)
        except TypeError:
            return super().compute_loss(raw_preds, batch)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        raw_preds = self._model_forward(batch)
        loss, terms = self.compute_loss(raw_preds, batch)
        bs = self._get_batch_size(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        for key, value in terms.items():
            if key == "loss":
                continue
            metric_value = self._coerce_log_value(value, ref=loss)
            self.log(f"train_{key}", metric_value, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self._append_history(
            self.train_loss_history,
            float(loss.detach().cpu().item()),
            max_points=self.max_step_history,
        )
        self._train_loss_sum += loss.detach().cpu().item() * bs
        self._train_loss_count += bs

        pdg_metrics = self._compute_node_pdg_batch_accuracy(raw_preds=raw_preds, batch=batch)
        if pdg_metrics is not None:
            self._append_pdg_batch_accuracy(split="train", batch_idx=batch_idx, metrics=pdg_metrics)
            overall = pdg_metrics.get("overall_accuracy")
            if isinstance(overall, float):
                self.log(
                    "train_pdg_accuracy_overall",
                    float(overall),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=bs,
                )
        endpoint_metrics = self._compute_endpoint_batch_mse(raw_preds=raw_preds, batch=batch)
        if endpoint_metrics is not None:
            self._append_endpoint_batch_mse(split="train", batch_idx=batch_idx, metrics=endpoint_metrics)
            overall_mse = endpoint_metrics.get("overall_mse")
            if isinstance(overall_mse, float):
                self.log(
                    "train_endpoint_mse_overall",
                    float(overall_mse),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=bs,
                )
        task_diag = self._compute_task_diagnostics_batch(raw_preds=raw_preds, batch=batch, loss_terms=terms)
        if task_diag is not None:
            self._append_task_diagnostics_batch(split="train", batch_idx=batch_idx, diagnostics=task_diag)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        if getattr(self, "_mc_dropout_in_val", True):
            self._enable_val_dropout()
        raw_preds = self._model_forward(batch)
        loss, terms = self.compute_loss(raw_preds, batch)
        bs = self._get_batch_size(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        for key, value in terms.items():
            if key == "loss":
                continue
            metric_value = self._coerce_log_value(value, ref=loss)
            self.log(f"val_{key}", metric_value, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self._append_history(
            self.val_loss_history,
            float(loss.detach().cpu().item()),
            max_points=self.max_step_history,
        )
        self._val_loss_sum += loss.detach().cpu().item() * bs
        self._val_loss_count += bs

        pdg_metrics = self._compute_node_pdg_batch_accuracy(raw_preds=raw_preds, batch=batch)
        if pdg_metrics is not None:
            self._append_pdg_batch_accuracy(split="val", batch_idx=batch_idx, metrics=pdg_metrics)
            overall = pdg_metrics.get("overall_accuracy")
            if isinstance(overall, float):
                self.log(
                    "val_pdg_accuracy_overall",
                    float(overall),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=bs,
                )
        endpoint_metrics = self._compute_endpoint_batch_mse(raw_preds=raw_preds, batch=batch)
        if endpoint_metrics is not None:
            self._append_endpoint_batch_mse(split="val", batch_idx=batch_idx, metrics=endpoint_metrics)
            overall_mse = endpoint_metrics.get("overall_mse")
            if isinstance(overall_mse, float):
                self.log(
                    "val_endpoint_mse_overall",
                    float(overall_mse),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    batch_size=bs,
                )
        task_diag = self._compute_task_diagnostics_batch(raw_preds=raw_preds, batch=batch, loss_terms=terms)
        if task_diag is not None:
            self._append_task_diagnostics_batch(split="val", batch_idx=batch_idx, diagnostics=task_diag)

    @staticmethod
    def _event_target(batch: Batch) -> torch.Tensor:
        for field in ("y_event", "y_graph", "y"):
            value = getattr(batch, field, None)
            if isinstance(value, torch.Tensor) and int(value.numel()) > 0:
                return value
        raise AttributeError("Batch is missing usable event-level target tensor (expected one of y_event/y_graph/y).")

    def primary_predictions(self, raw_preds: Any) -> torch.Tensor:
        self._last_token_batch = None
        self._last_token_valid = None

        if isinstance(raw_preds, Mapping):
            logits = raw_preds.get("unified_event_logits")
            if not isinstance(logits, torch.Tensor):
                maybe_main = raw_preds.get("main")
                if isinstance(maybe_main, torch.Tensor):
                    logits = maybe_main
            if not isinstance(logits, torch.Tensor):
                return super().primary_predictions(raw_preds)

            token_batch = raw_preds.get("unified_token_batch")
            token_valid = raw_preds.get("unified_token_valid")
            if isinstance(token_batch, torch.Tensor):
                token_batch = token_batch.to(dtype=torch.long)
                if token_batch.dim() != 1:
                    token_batch = token_batch.view(-1)
                if logits.dim() > 1:
                    logits = logits.view(logits.shape[0], -1)
                if logits.shape[0] == token_batch.shape[0]:
                    if isinstance(token_valid, torch.Tensor):
                        keep = token_valid.to(dtype=torch.bool).view(-1)
                        if keep.shape[0] == logits.shape[0]:
                            logits = logits[keep]
                            token_batch = token_batch[keep]
                            self._last_token_valid = keep
                    self._last_token_batch = token_batch
            return logits

        return super().primary_predictions(raw_preds)

    def primary_target(self, batch: Batch, preds: torch.Tensor) -> torch.Tensor:
        event_target = self._event_target(batch)
        token_batch = self._last_token_batch
        if isinstance(token_batch, torch.Tensor) and int(token_batch.numel()) == int(preds.shape[0]):
            idx = token_batch.to(device=event_target.device, dtype=torch.long)
            if int(idx.numel()) > 0:
                if int(idx.min().item()) < 0 or int(idx.max().item()) >= int(event_target.shape[0]):
                    raise ValueError(
                        f"Token batch indices out of range for event targets: "
                        f"max_idx={int(idx.max().item())}, num_events={int(event_target.shape[0])}"
                    )
            target = event_target.index_select(0, idx)
            if target.dim() == 1 and preds.dim() == 2:
                target = target.unsqueeze(1)
            return target
        return super().primary_target(batch, preds)

    def on_before_optimizer_step(self, optimizer: optim.Optimizer) -> None:
        """Fail fast on non-finite gradients before parameters are updated."""
        for name, param in self.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if not bool(torch.isfinite(grad).all().item()):
                flat = grad.detach().reshape(-1)
                finite = torch.isfinite(flat)
                finite_count = int(finite.sum().item())
                nan_count = int(torch.isnan(flat).sum().item())
                inf_count = int(torch.isinf(flat).sum().item())
                min_v = float(flat[finite].min().item()) if finite_count > 0 else float("nan")
                max_v = float(flat[finite].max().item()) if finite_count > 0 else float("nan")
                raise RuntimeError(
                    "[purity_module] non-finite gradient detected before optimizer step "
                    f"for parameter={name}\n"
                    f"shape={tuple(grad.shape)} numel={int(flat.numel())} "
                    f"finite={finite_count} nan={nan_count} inf={inf_count} "
                    f"min={min_v:.6g} max={max_v:.6g}"
                )
        return super().on_before_optimizer_step(optimizer)

    def set_optimizer_hyperparams(self, *, lr: float | None = None, weight_decay: float | None = None) -> None:
        if lr is not None:
            lr_f = float(lr)
            if not math.isfinite(lr_f) or lr_f <= 0.0:
                raise ValueError("lr must be finite and > 0.")
            self.lr = lr_f
        if weight_decay is not None:
            wd_f = float(weight_decay)
            if not math.isfinite(wd_f) or wd_f < 0.0:
                raise ValueError("weight_decay must be finite and >= 0.")
            self.weight_decay = wd_f

    def configure_optimizers(self):
        param_groups = self._build_optimizer_param_groups()
        if param_groups is None:
            optimizer = self.optimizer_cls(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = self.optimizer_cls(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler_step_size is None:
            return optimizer
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _build_optimizer_param_groups(self) -> list[dict[str, Any]] | None:
        raw_groups = list(self.optimizer_param_groups or [])
        if len(raw_groups) == 0:
            return None

        named_params = list(self.named_parameters())
        compiled: list[dict[str, Any]] = []
        seen: set[int] = set()

        for idx, raw in enumerate(raw_groups):
            if not isinstance(raw, Mapping):
                continue
            group = dict(raw)
            patterns_raw = group.get("patterns")
            if not isinstance(patterns_raw, list) or len(patterns_raw) == 0:
                continue
            patterns = [str(p).strip() for p in patterns_raw if str(p).strip() != ""]
            if len(patterns) == 0:
                continue
            regexes = [re.compile(p) for p in patterns]

            params = []
            for name, param in named_params:
                if not bool(param.requires_grad):
                    continue
                pid = id(param)
                if pid in seen:
                    continue
                candidates = [name]
                if name.startswith("model."):
                    candidates.append(name[len("model.") :])
                if any(r.search(candidate) for r in regexes for candidate in candidates):
                    params.append(param)
                    seen.add(pid)
            if len(params) == 0:
                continue

            pg: dict[str, Any] = {"params": params}
            if group.get("lr") is not None:
                pg["lr"] = float(group["lr"])
            elif group.get("lr_scale") is not None:
                pg["lr"] = float(self.lr) * float(group["lr_scale"])
            if group.get("weight_decay") is not None:
                pg["weight_decay"] = float(group["weight_decay"])
            compiled.append(pg)

        base_params = [p for _, p in named_params if bool(p.requires_grad) and id(p) not in seen]
        if len(base_params) > 0:
            compiled.append({"params": base_params})
        if len(compiled) == 0:
            return None
        return compiled

    def _compute_node_pdg_batch_accuracy(self, *, raw_preds: Any, batch: Batch) -> dict[str, Any] | None:
        if not isinstance(raw_preds, Mapping):
            return None
        logits = raw_preds.get("atar_node_pdg")
        if not isinstance(logits, torch.Tensor) or logits.dim() != 2 or int(logits.numel()) <= 0:
            return None

        target = getattr(batch, "atar_node_pdg_target", None)
        if not isinstance(target, torch.Tensor):
            return None
        target = target.to(device=logits.device, dtype=torch.float32)
        if target.dim() != 2 or int(target.numel()) <= 0:
            return None

        x_all = getattr(batch, "x", None)
        if isinstance(x_all, torch.Tensor) and x_all.dim() == 2 and int(x_all.shape[0]) == int(target.shape[0]):
            is_lyso_all = x_all[:, 7] > 0.5
            is_atar_all = ~is_lyso_all
            target = target[is_atar_all]

        n_rows = min(int(logits.shape[0]), int(target.shape[0]))
        n_cols = min(int(logits.shape[1]), int(target.shape[1]))
        if n_rows <= 0 or n_cols <= 0:
            return None
        logits = logits[:n_rows, :n_cols]
        target = target[:n_rows, :n_cols]

        pred = (torch.sigmoid(logits.detach()) >= 0.5).to(dtype=torch.float32)
        truth = (target.detach() >= 0.5).to(dtype=torch.float32)
        eq = (pred == truth).to(dtype=torch.float32)
        if int(eq.numel()) <= 0:
            return None

        class_acc_vec = eq.mean(dim=0)
        class_names = list(self.pdg_class_names[:n_cols])
        if len(class_names) < n_cols:
            for idx in range(len(class_names), n_cols):
                class_names.append(f"class_{idx}")
        class_accuracy = {
            str(name): float(class_acc_vec[idx].cpu().item())
            for idx, name in enumerate(class_names)
        }
        return {
            "overall_accuracy": float(eq.mean().cpu().item()),
            "class_accuracy": class_accuracy,
            "num_samples": int(n_rows),
        }

    def _append_pdg_batch_accuracy(self, *, split: str, batch_idx: int, metrics: Mapping[str, Any]) -> None:
        if str(split) == "train":
            history = self.train_pdg_batch_accuracy_history
        else:
            history = self.val_pdg_batch_accuracy_history
        entry: dict[str, Any] = {
            "batch_idx": int(batch_idx),
            "global_batch_index": int(len(history)),
            "overall_accuracy": float(metrics.get("overall_accuracy", 0.0)),
            "class_accuracy": {
                str(k): float(v)
                for k, v in dict(metrics.get("class_accuracy") or {}).items()
            },
            "num_samples": int(metrics.get("num_samples", 0)),
        }
        phase = getattr(self, "_staged_training_current_phase", None)
        if isinstance(phase, Mapping):
            if phase.get("index") is not None:
                entry["phase_index"] = int(phase["index"])
            if phase.get("name") is not None:
                entry["phase_name"] = str(phase["name"])
        history.append(entry)

    @staticmethod
    def _compute_endpoint_batch_mse(*, raw_preds: Any, batch: Batch) -> dict[str, Any] | None:
        if not isinstance(raw_preds, Mapping):
            return None
        preds = raw_preds.get("atar_endpoints")
        if not isinstance(preds, torch.Tensor) or int(preds.numel()) <= 0:
            return None

        start_target = getattr(batch, "atar_slice_start_target", None)
        stop_target = getattr(batch, "atar_slice_stop_target", None)
        if not isinstance(start_target, torch.Tensor) or not isinstance(stop_target, torch.Tensor):
            return None
        if start_target.dim() != 2 or stop_target.dim() != 2:
            return None
        if int(start_target.size(1)) < 3 or int(stop_target.size(1)) < 3:
            return None

        preds = preds.to(dtype=torch.float32)
        pred_median = preds[..., 1] if int(preds.dim()) == 4 and int(preds.size(-1)) >= 2 else None
        if pred_median is None:
            return None
        if pred_median.dim() != 3 or int(pred_median.size(1)) < 2 or int(pred_median.size(2)) < 3:
            return None

        pred_start = pred_median[:, 0, :3]
        pred_stop = pred_median[:, 1, :3]

        start_target = start_target.to(device=pred_start.device, dtype=torch.float32)[:, :3]
        stop_target = stop_target.to(device=pred_stop.device, dtype=torch.float32)[:, :3]

        n_rows = min(int(pred_start.size(0)), int(start_target.size(0)), int(pred_stop.size(0)), int(stop_target.size(0)))
        if n_rows <= 0:
            return None
        pred_start = pred_start[:n_rows]
        pred_stop = pred_stop[:n_rows]
        start_target = start_target[:n_rows]
        stop_target = stop_target[:n_rows]

        pred_xyz = torch.stack([pred_start, pred_stop], dim=1) * 10.0
        tar_xyz = torch.stack([start_target, stop_target], dim=1) * 10.0

        diff = pred_xyz - tar_xyz
        sq = diff.pow(2)
        mse_total = sq.mean()
        mse_start = sq[:, 0, :].mean()
        mse_stop = sq[:, 1, :].mean()
        mse_x = sq[:, :, 0].mean()
        mse_y = sq[:, :, 1].mean()
        mse_z = sq[:, :, 2].mean()

        class_mse: dict[str, float] = {}
        class_counts: dict[str, int] = {}
        class_names = ("pion", "muon", "mip")
        slice_pdg_target = getattr(batch, "atar_slice_pdg_target", None)
        if isinstance(slice_pdg_target, torch.Tensor) and slice_pdg_target.dim() == 2:
            slice_pdg_target = slice_pdg_target.to(device=pred_xyz.device, dtype=torch.float32)
            n_cls_rows = min(int(slice_pdg_target.size(0)), int(n_rows))
            n_cls = min(int(slice_pdg_target.size(1)), int(len(class_names)))
            if n_cls_rows > 0 and n_cls > 0:
                cls_target = slice_pdg_target[:n_cls_rows, :n_cls]
                cls_sq = sq[:n_cls_rows]
                for idx in range(n_cls):
                    name = str(class_names[idx])
                    mask = cls_target[:, idx] > 0.5
                    count = int(mask.sum().item())
                    class_counts[name] = count
                    if count > 0:
                        class_mse[name] = float(cls_sq[mask].mean().detach().cpu().item())
                    else:
                        class_mse[name] = float("nan")

        return {
            "overall_mse": float(mse_total.detach().cpu().item()),
            "start_mse": float(mse_start.detach().cpu().item()),
            "stop_mse": float(mse_stop.detach().cpu().item()),
            "axis_mse": {
                "x": float(mse_x.detach().cpu().item()),
                "y": float(mse_y.detach().cpu().item()),
                "z": float(mse_z.detach().cpu().item()),
            },
            "class_mse": class_mse,
            "class_counts": class_counts,
            "num_samples": int(n_rows),
        }

    def _append_endpoint_batch_mse(self, *, split: str, batch_idx: int, metrics: Mapping[str, Any]) -> None:
        if str(split) == "train":
            history = self.train_endpoint_mse_batch_history
        else:
            history = self.val_endpoint_mse_batch_history
        axis_raw = metrics.get("axis_mse")
        axis = dict(axis_raw) if isinstance(axis_raw, Mapping) else {}
        class_raw = metrics.get("class_mse")
        class_mse = dict(class_raw) if isinstance(class_raw, Mapping) else {}
        counts_raw = metrics.get("class_counts")
        class_counts = dict(counts_raw) if isinstance(counts_raw, Mapping) else {}
        entry: dict[str, Any] = {
            "batch_idx": int(batch_idx),
            "global_batch_index": int(len(history)),
            "overall_mse": float(metrics.get("overall_mse", 0.0)),
            "start_mse": float(metrics.get("start_mse", 0.0)),
            "stop_mse": float(metrics.get("stop_mse", 0.0)),
            "axis_mse": {
                "x": float(axis.get("x", 0.0)),
                "y": float(axis.get("y", 0.0)),
                "z": float(axis.get("z", 0.0)),
            },
            "class_mse": {str(k): float(v) for k, v in class_mse.items()},
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "num_samples": int(metrics.get("num_samples", 0)),
        }
        phase = getattr(self, "_staged_training_current_phase", None)
        if isinstance(phase, Mapping):
            if phase.get("index") is not None:
                entry["phase_index"] = int(phase["index"])
            if phase.get("name") is not None:
                entry["phase_name"] = str(phase["name"])
        history.append(entry)

    @staticmethod
    def _to_float_scalar(value: Any) -> float | None:
        if isinstance(value, torch.Tensor):
            if int(value.numel()) <= 0:
                return None
            return float(value.detach().float().mean().cpu().item())
        if isinstance(value, (float, int)):
            return float(value)
        return None

    @staticmethod
    def _truth_positron_time_proxy(batch: Batch, *, device: torch.device) -> torch.Tensor | None:
        x = getattr(batch, "x", None)
        b = getattr(batch, "batch", None)
        if not isinstance(x, torch.Tensor) or not isinstance(b, torch.Tensor):
            return None
        is_atar = (x[:, 5] > 0.5) | (x[:, 6] > 0.5)
        n_graphs = int(b.max().item()) + 1 if int(b.numel()) > 0 else 0
        if n_graphs <= 0 or int(is_atar.sum().item()) <= 0:
            return torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=device)

        trig = getattr(batch, "is_trigger_target", None)
        node = getattr(batch, "atar_node_pdg_target", None)
        if not isinstance(trig, torch.Tensor) or not isinstance(node, torch.Tensor):
            return torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=device)
        if int(node.dim()) != 2 or int(node.size(1)) < 3:
            return torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=device)

        trig = trig.view(-1).to(device=device, dtype=torch.float32)
        node = node.to(device=device, dtype=torch.float32)
        b = b.to(device=device)
        x = x.to(device=device, dtype=torch.float32)
        trig_atar = trig[is_atar] if int(trig.numel()) == int(x.size(0)) else trig
        node_atar = node[is_atar] if int(node.size(0)) == int(x.size(0)) else node
        n = min(int(node_atar.size(0)), int(trig_atar.numel()), int(is_atar.sum().item()))
        if n <= 0:
            return torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=device)

        atar_times = x[is_atar, 4][:n]
        atar_batch = b[is_atar][:n]
        mask = (trig_atar[:n] > 0.5) & (node_atar[:n, 2] > 0.5)

        num = torch.zeros((n_graphs,), dtype=torch.float32, device=device)
        den = torch.zeros((n_graphs,), dtype=torch.float32, device=device)
        num.index_add_(0, atar_batch, atar_times * mask.float())
        den.index_add_(0, atar_batch, mask.float())
        out = torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=device)
        valid = den > 0.5
        out[valid] = num[valid] / den[valid]
        return out

    def _compute_task_diagnostics_batch(
        self,
        *,
        raw_preds: Any,
        batch: Batch,
        loss_terms: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(raw_preds, Mapping):
            return None

        diagnostics: dict[str, Any] = {"losses": {}, "metrics": {}}

        losses: dict[str, float] = {}
        for key, value in dict(loss_terms or {}).items():
            name = str(key)
            if name == "loss":
                continue
            if not (
                name.startswith("loss_")
                or name.startswith("L_")
                or name.startswith("end_")
                or name.startswith("lyso_")
            ):
                continue
            scalar = self._to_float_scalar(value)
            if scalar is None:
                continue
            losses[name] = float(scalar)
        if losses:
            diagnostics["losses"] = losses

        metrics: dict[str, float] = {}

        # Trigger-slice BCE metric
        trig_logits = raw_preds.get("atar_trigger_logits")
        trig_target = getattr(batch, "atar_slice_trigger_target", None)
        if isinstance(trig_logits, torch.Tensor) and isinstance(trig_target, torch.Tensor):
            n = min(int(trig_logits.numel()), int(trig_target.numel()))
            if n > 0:
                l = F.binary_cross_entropy_with_logits(
                    trig_logits.view(-1)[:n],
                    trig_target.to(device=trig_logits.device, dtype=torch.float32).view(-1)[:n],
                )
                metrics["trigger_slice_bce"] = float(l.detach().cpu().item())

        # Slice-multi BCE metric
        multi_logits = raw_preds.get("atar_slice_multi")
        multi_target = getattr(batch, "atar_slice_multi_target", None)
        if isinstance(multi_logits, torch.Tensor) and isinstance(multi_target, torch.Tensor):
            n = min(int(multi_logits.numel()), int(multi_target.numel()))
            if n > 0:
                l = F.binary_cross_entropy_with_logits(
                    multi_logits.view(-1)[:n],
                    multi_target.to(device=multi_logits.device, dtype=torch.float32).view(-1)[:n],
                )
                metrics["slice_multi_bce"] = float(l.detach().cpu().item())

        # Positron angle metrics
        has_pos = getattr(batch, "has_trigger_positron", None)
        pos_dir = raw_preds.get("atar_positron_dir")
        angle_target = getattr(batch, "atar_angle_target", None)
        angle_batch = getattr(batch, "atar_batch", None)
        if (
            isinstance(pos_dir, torch.Tensor)
            and isinstance(has_pos, torch.Tensor)
            and isinstance(angle_target, torch.Tensor)
            and isinstance(angle_batch, torch.Tensor)
            and int(pos_dir.dim()) == 2
            and int(pos_dir.size(1)) >= 3
            and int(angle_target.dim()) == 2
            and int(angle_target.size(1)) >= 3
        ):
            try:
                from torch_geometric.utils import scatter

                n_graphs = int(getattr(batch, "num_graphs", int(pos_dir.size(0))))
                tar_dir = scatter(
                    angle_target.to(device=pos_dir.device, dtype=torch.float32)[:, :3],
                    angle_batch.to(device=pos_dir.device, dtype=torch.long),
                    dim=0,
                    dim_size=n_graphs,
                    reduce="mean",
                )
                n = min(int(pos_dir.size(0)), int(tar_dir.size(0)), int(has_pos.numel()))
                if n > 0:
                    pos_mask = has_pos.to(device=pos_dir.device, dtype=torch.bool)[:n]
                    if bool(pos_mask.any().item()):
                        pred_dirs = pos_dir[:n][pos_mask]
                        tar_dirs = tar_dir[:n][pos_mask]
                        cos_err = (1.0 - F.cosine_similarity(pred_dirs, tar_dirs, dim=1)).mean()
                        metrics["positron_angle_cosine_error"] = float(cos_err.detach().cpu().item())

                        pred_theta = torch.acos(
                            pred_dirs[:, 2].div(pred_dirs.norm(dim=1).clamp(min=1e-9)).clamp(-1.0, 1.0)
                        )
                        truth_theta = torch.acos(tar_dirs[:, 2].clamp(-1.0, 1.0))
                        theta_mae = (truth_theta - pred_theta).abs().mean().rad2deg()
                        metrics["positron_theta_mae_deg"] = float(theta_mae.detach().cpu().item())
            except Exception:
                pass

        # Positron time metric (proxy truth)
        pt_pred = raw_preds.get("positron_time_per_graph")
        if isinstance(pt_pred, torch.Tensor):
            pt_truth = self._truth_positron_time_proxy(batch, device=pt_pred.device)
            if isinstance(pt_truth, torch.Tensor):
                n = min(int(pt_pred.numel()), int(pt_truth.numel()))
                if n > 0:
                    pred_t = pt_pred.view(-1)[:n].to(dtype=torch.float32)
                    truth_t = pt_truth.view(-1)[:n].to(dtype=torch.float32)
                    valid = torch.isfinite(pred_t) & torch.isfinite(truth_t)
                    if bool(valid.any().item()):
                        # Model/targets store ATAR time normalized by 500.0.
                        mae_ns = (pred_t[valid] - truth_t[valid]).abs().mean() * 500.0
                        metrics["positron_time_mae_proxy_ns"] = float(mae_ns.detach().cpu().item())

        # Event-builder BCE proxy on token logits vs mapped event truth
        ev_logits = raw_preds.get("unified_event_logits")
        tok_batch = raw_preds.get("unified_token_batch")
        if isinstance(ev_logits, torch.Tensor) and isinstance(tok_batch, torch.Tensor):
            y_event = None
            for field in ("y_event", "y_graph", "y"):
                value = getattr(batch, field, None)
                if isinstance(value, torch.Tensor) and int(value.numel()) > 0:
                    y_event = value.to(device=ev_logits.device, dtype=torch.float32).view(-1)
                    break
            if isinstance(y_event, torch.Tensor):
                logits = ev_logits.view(-1)
                idx = tok_batch.to(device=ev_logits.device, dtype=torch.long).view(-1)
                n = min(int(logits.numel()), int(idx.numel()))
                if n > 0:
                    logits = logits[:n]
                    idx = idx[:n]
                    valid = (idx >= 0) & (idx < int(y_event.numel()))
                    if bool(valid.any().item()):
                        l = F.binary_cross_entropy_with_logits(
                            logits[valid],
                            y_event[idx[valid]],
                        )
                        metrics["event_builder_bce_proxy"] = float(l.detach().cpu().item())

        if metrics:
            diagnostics["metrics"] = metrics

        if not diagnostics["losses"] and not diagnostics["metrics"]:
            return None
        return diagnostics

    def _append_task_diagnostics_batch(self, *, split: str, batch_idx: int, diagnostics: Mapping[str, Any]) -> None:
        if str(split) == "train":
            history = self.train_task_diagnostics_batch_history
        else:
            history = self.val_task_diagnostics_batch_history

        losses_raw = diagnostics.get("losses")
        metrics_raw = diagnostics.get("metrics")
        losses = dict(losses_raw) if isinstance(losses_raw, Mapping) else {}
        metrics = dict(metrics_raw) if isinstance(metrics_raw, Mapping) else {}
        entry: dict[str, Any] = {
            "batch_idx": int(batch_idx),
            "global_batch_index": int(len(history)),
            "losses": {str(k): float(v) for k, v in losses.items()},
            "metrics": {str(k): float(v) for k, v in metrics.items()},
        }
        phase = getattr(self, "_staged_training_current_phase", None)
        if isinstance(phase, Mapping):
            if phase.get("index") is not None:
                entry["phase_index"] = int(phase["index"])
            if phase.get("name") is not None:
                entry["phase_name"] = str(phase["name"])
        history.append(entry)

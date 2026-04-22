from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Dict

import torch
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
        # Omar parity: default optimizer is Adam (not AdamW).
        if optimizer_cls is None:
            optimizer_cls = optim.Adam
        super().__init__(*args, optimizer_cls=optimizer_cls, **kwargs)
        self._last_token_batch: torch.Tensor | None = None
        self._last_token_valid: torch.Tensor | None = None
        self._task_weights: dict[str, float] | None = None
        self.optimizer_param_groups: list[dict[str, Any]] = [
            dict(item) for item in list(optimizer_param_groups or [])
        ]
        self.pdg_class_names: tuple[str, ...] = ("pion", "muon", "mip")
        self.train_pdg_batch_accuracy_history: list[dict[str, Any]] = []
        self.val_pdg_batch_accuracy_history: list[dict[str, Any]] = []

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
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
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

from __future__ import annotations

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

    def __init__(self, *args, optimizer_cls: type[optim.Optimizer] | None = None, **kwargs):
        # Omar parity: default optimizer is Adam (not AdamW).
        if optimizer_cls is None:
            optimizer_cls = optim.Adam
        super().__init__(*args, optimizer_cls=optimizer_cls, **kwargs)
        self._last_token_batch: torch.Tensor | None = None
        self._last_token_valid: torch.Tensor | None = None
        self._task_weights: dict[str, float] | None = None

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

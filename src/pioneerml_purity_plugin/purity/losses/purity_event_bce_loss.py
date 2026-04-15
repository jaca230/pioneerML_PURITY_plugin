from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from pioneerml.integration.pytorch.losses.base_loss import BaseLoss
from pioneerml.integration.pytorch.losses.factory.registry import REGISTRY as LOSS_REGISTRY


@LOSS_REGISTRY.register("purity_event_bce")
@LOSS_REGISTRY.register("purity_bce")
class PurityEventBCELoss(BaseLoss):
    """Binary event-level BCE loss with optional fixed positive-class weighting."""

    def __init__(self, *, pos_weight: float | None = None) -> None:
        super().__init__()
        if pos_weight is None:
            self.pos_weight = None
        else:
            val = float(pos_weight)
            if val <= 0.0:
                raise ValueError("pos_weight must be > 0 when provided.")
            self.pos_weight = val

    @staticmethod
    def _resolve_event_target(target_like: object) -> torch.Tensor:
        if isinstance(target_like, torch.Tensor):
            return target_like
        for field in ("y_event", "y_graph", "y"):
            value = getattr(target_like, field, None)
            if isinstance(value, torch.Tensor) and int(value.numel()) > 0:
                return value
        raise AttributeError("PURITY loss expected event-level target tensor (one of: y_event, y_graph, y).")

    def _bce_logits(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if preds.shape != target.shape:
            raise ValueError(
                f"purity_event_bce requires matching preds/target shapes, got {tuple(preds.shape)} vs {tuple(target.shape)}"
            )

        valid = torch.isfinite(preds) & torch.isfinite(target)
        if not bool(valid.any()):
            return preds.sum() * 0.0

        preds_valid = preds[valid]
        target_valid = target[valid].to(dtype=preds_valid.dtype)

        if self.pos_weight is None:
            return F.binary_cross_entropy_with_logits(preds_valid, target_valid)

        pos_weight = torch.tensor(float(self.pos_weight), device=preds_valid.device, dtype=preds_valid.dtype)
        return F.binary_cross_entropy_with_logits(preds_valid, target_valid, pos_weight=pos_weight)

    def forward(self, preds: torch.Tensor | Mapping[str, object], target: torch.Tensor | object) -> torch.Tensor:
        if isinstance(preds, Mapping):
            logits = preds.get("unified_event_logits")
            if not isinstance(logits, torch.Tensor):
                maybe_main = preds.get("main")
                if isinstance(maybe_main, torch.Tensor):
                    logits = maybe_main
            if not isinstance(logits, torch.Tensor):
                raise TypeError("PURITY loss expected mapping output to contain tensor 'unified_event_logits' or 'main'.")

            event_target = self._resolve_event_target(target).to(device=logits.device)
            token_batch = preds.get("unified_token_batch")
            token_valid = preds.get("unified_token_valid")

            if isinstance(token_batch, torch.Tensor):
                token_batch_idx = token_batch.to(device=event_target.device, dtype=torch.long)
                if token_batch_idx.dim() != 1:
                    token_batch_idx = token_batch_idx.view(-1)
                if logits.shape[0] != token_batch_idx.shape[0]:
                    raise ValueError(
                        f"unified_event_logits/token_batch length mismatch: {tuple(logits.shape)} vs {tuple(token_batch_idx.shape)}"
                    )
                if int(token_batch_idx.numel()) > 0:
                    if int(token_batch_idx.min().item()) < 0 or int(token_batch_idx.max().item()) >= int(event_target.shape[0]):
                        raise ValueError(
                            f"unified_token_batch contains out-of-range graph ids for y_event/y_graph size {int(event_target.shape[0])}."
                        )
                aligned_target = event_target.index_select(0, token_batch_idx)
                aligned_logits = logits
                if isinstance(token_valid, torch.Tensor):
                    keep = token_valid.to(device=aligned_target.device, dtype=torch.bool).view(-1)
                    if keep.shape[0] == aligned_target.shape[0]:
                        aligned_target = aligned_target[keep]
                        aligned_logits = aligned_logits[keep]
                return self._bce_logits(aligned_logits, aligned_target)

            return self._bce_logits(logits, event_target)

        if not isinstance(preds, torch.Tensor):
            raise TypeError("PURITY loss expected tensor predictions or mapping predictions.")
        if not isinstance(target, torch.Tensor):
            target = self._resolve_event_target(target)
        return self._bce_logits(preds, target)

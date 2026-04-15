from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from pioneerml.integration.pytorch.losses.base_loss import BaseLoss
from pioneerml.integration.pytorch.losses.factory.registry import REGISTRY as LOSS_REGISTRY

from .utils.unified_training_components import PURITYLoss, format_targets_from_batch


@LOSS_REGISTRY.register("purity_unified")
@LOSS_REGISTRY.register("purity_multitask")
class PurityUnifiedLoss(BaseLoss):
    """Strict wrapper around the unified_reco `PURITYLoss` implementation."""

    DEFAULT_TASK_WEIGHTS = {
        "w_atar_slice_multi": 0.1,
        "w_node_pdg": 0.5,
        "w_atar_edge": 0.25,
        "w_node_trigger": 0.0,
        "w_slice_pdg": 1.0,
        "w_atar_trigger_slice": 0.0,
        "w_pion_kinematics": 0.0,
        "w_endpoints": 0.1,
        "w_positron_angle": 0.0,
        "w_lyso_condensation": 0.25,
        "w_positron_energy": 0.0,
        "w_event_builder": 0.0,
    }

    def __init__(self, **config: Any):
        super().__init__()
        self.config = dict(self.DEFAULT_TASK_WEIGHTS)
        self.config.update(dict(config or {}))
        self.criterion = PURITYLoss(config=dict(self.config))

    def set_task_weights(self, task_weights: Mapping[str, float] | None) -> None:
        self.config = dict(self.DEFAULT_TASK_WEIGHTS)
        if task_weights is not None:
            for key, value in dict(task_weights).items():
                self.config[str(key)] = float(value)
        self.criterion.config = dict(self.config)

    def forward(self, preds: torch.Tensor | Mapping[str, Any], target: torch.Tensor | Any):
        if not isinstance(preds, Mapping):
            if not isinstance(preds, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise TypeError("purity_unified expects mapping predictions with batch target context.")
            return F.binary_cross_entropy_with_logits(preds, target)

        batch = target
        targets = format_targets_from_batch(batch)
        total_loss, loss_dict = self.criterion(preds, targets, batch=batch)

        # Ensure framework receives a dict with explicit total-loss key.
        out_dict = dict(loss_dict)
        if "loss_total" not in out_dict:
            out_dict["loss_total"] = total_loss
        return total_loss, out_dict

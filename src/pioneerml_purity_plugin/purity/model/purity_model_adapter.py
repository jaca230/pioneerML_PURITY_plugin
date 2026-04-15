"""Framework adapter layer around the core PURITY hybrid model."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import warnings

import torch
import torch.nn as nn
from torch_geometric.data import Data

from pioneerml.integration.pytorch.models.architectures.factory.registry import REGISTRY as ARCHITECTURE_REGISTRY
from pioneerml.integration.pytorch.models.architectures.graph.transformer.classifiers.base_graph_classifier_model import (
    BaseGraphClassifierModel,
)

from .purity_hybrid_model import PurityHybridModel


class _PurityExportAdapter(nn.Module):
    """TorchScript export adapter preserving the framework export signature."""

    def __init__(self, model: "PurityModel"):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        _ = edge_index
        _ = edge_attr
        outputs = self.model.forward_tensors(x, batch, task_weights=None)
        logits = outputs.get("unified_event_logits")
        if isinstance(logits, torch.Tensor):
            return logits
        return x.new_zeros((0, 1))


@ARCHITECTURE_REGISTRY.register("purity")
@ARCHITECTURE_REGISTRY.register("purity_model")
class PurityModel(BaseGraphClassifierModel):
    """
    Adapter wrapper for the core `PurityHybridModel`.

    The internal implementation mirrors:
    `deprecated/omar_pioneerML/unified_reco/models.py`
    """

    def __init__(
        self,
        node_dim: int = 10,
        edge_dim: int = 11,
        graph_dim: int = 0,
        hidden: int = 150,
        heads: int = 5,
        layers: int = 3,
        num_blocks: int | None = None,
        dropout: float = 0.05,
        num_pdg_classes: int = 3,
    ):
        resolved_blocks = int(layers) if num_blocks is None else int(num_blocks)
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            graph_dim=graph_dim,
            hidden=hidden,
            dropout=dropout,
        )
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.num_blocks = int(resolved_blocks)
        self.num_pdg_classes = int(num_pdg_classes)

        self.impl = PurityHybridModel(
            hidden_dim=int(hidden),
            num_blocks=int(resolved_blocks),
            heads=int(heads),
            dropout=float(dropout),
            num_pdg_classes=int(num_pdg_classes),
        )

    @staticmethod
    def _resolve_batch_input(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x = getattr(data, "x_node", None)
        if not isinstance(x, torch.Tensor):
            x = getattr(data, "x", None)
        if not isinstance(x, torch.Tensor):
            raise AttributeError("PurityModel requires node features in one of: x_node, x.")

        batch = getattr(data, "node_graph_id", None)
        if not isinstance(batch, torch.Tensor):
            batch = getattr(data, "batch", None)
        if not isinstance(batch, torch.Tensor):
            raise AttributeError("PurityModel requires node graph ids in one of: node_graph_id, batch.")

        return x, batch

    @torch.jit.ignore
    def forward(
        self,
        data_or_x: Data | torch.Tensor,
        batch: torch.Tensor | None = None,
        task_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        if isinstance(data_or_x, Data):
            x_node, node_graph_id = self._resolve_batch_input(data_or_x)
            return self.impl(x_node, node_graph_id, task_weights=task_weights)

        if batch is None:
            raise ValueError("PurityModel.forward requires `batch` when called with tensor inputs.")
        return self.impl(data_or_x, batch, task_weights=task_weights)

    def forward_tensors(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        task_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return self.impl(x, batch, task_weights=task_weights)

    @staticmethod
    def extract_event_logits(raw_output: Any) -> torch.Tensor:
        if isinstance(raw_output, torch.Tensor):
            return raw_output
        if isinstance(raw_output, (tuple, list)) and len(raw_output) > 0 and isinstance(raw_output[0], torch.Tensor):
            return raw_output[0]
        if isinstance(raw_output, Mapping):
            value = raw_output.get("unified_event_logits")
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError("Unable to extract event logits from PURITY model output.")

    def export_torchscript(
        self,
        path: str | Path | None,
        *,
        prefer_cuda: bool = True,
        strict: bool = False,
    ) -> torch.jit.ScriptModule:
        _ = strict
        device = torch.device("cuda") if prefer_cuda and torch.cuda.is_available() else torch.device("cpu")
        self.eval()
        self.to(device)
        scriptable = _PurityExportAdapter(self)
        try:
            scripted = torch.jit.script(scriptable)
        except Exception:
            # Fallback for environments where scripting rejects dynamic control flow.
            example_x = torch.tensor(
                [
                    [0.1, 0.0, 0.1, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0],
                    [0.2, 0.0, 0.2, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0, 10.0],
                    [0.0, 0.1, 0.2, 0.4, 0.2, 0.0, 1.0, 0.0, 0.0, 11.0],
                    [0.3, 0.2, 0.3, 0.6, 0.3, 0.0, 0.0, 1.0, 1.0, 12.0],
                ],
                dtype=torch.float32,
                device=device,
            )
            example_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            example_edge_attr = torch.zeros((0, int(self.edge_dim)), dtype=torch.float32, device=device)
            example_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings(
                    "ignore",
                    message=".*Converting a tensor to a Python.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*torch.tensor results are registered as constants in the trace.*",
                    category=UserWarning,
                )
                scripted = torch.jit.trace(
                    scriptable,
                    (example_x, example_edge_index, example_edge_attr, example_batch),
                    strict=False,
                    check_trace=False,
                )
        if path is not None:
            scripted.save(str(path))
        return scripted

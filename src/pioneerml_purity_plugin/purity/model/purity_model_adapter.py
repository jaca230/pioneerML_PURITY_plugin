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
    ) -> dict[str, torch.Tensor]:
        _ = edge_index
        _ = edge_attr
        outputs = self.model.forward_tensors(x, batch, task_weights=None)
        logits = outputs.get("unified_event_logits")
        if logits is None:
            logits = x.new_zeros((0, 1))

        token_batch = outputs.get("unified_token_batch")
        if token_batch is None:
            token_batch = torch.zeros((int(logits.size(0)),), dtype=torch.long, device=logits.device)
        token_batch = token_batch.to(dtype=torch.long)
        # Trace-safe normalization: keep token_batch on the same compact axis as logits.
        token_batch = token_batch[: int(logits.size(0))]

        # Trace-safe compact validity mask.
        token_valid = torch.ones((int(logits.size(0)),), dtype=torch.bool, device=logits.device)

        payload: dict[str, torch.Tensor] = {
            "unified_event_logits": logits,
            "unified_token_batch": token_batch,
            "unified_token_valid": token_valid,
            # Keep "main" for generic writer/model-handle paths that look for a default output.
            "main": logits,
        }
        for k in (
            "summary_accepted",
            "summary_positron_energy",
            "summary_positron_time",
            "summary_positron_polar_angle",
        ):
            v = outputs.get(k)
            if isinstance(v, torch.Tensor):
                payload[k] = v
        return payload


def _build_trace_example(
    *,
    device: torch.device,
    edge_dim: int,
    max_graphs: int = 128,
    max_slices: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a broad synthetic batch for trace fallback.

    The core model uses Python `int(...)` conversions on `batch.max()` and `slice_id.max()`.
    In traced mode, those values become constants, so we intentionally trace with a
    wide representative batch to avoid shape/index under-allocation at inference time.
    """
    rows: list[list[float]] = []
    batch_ids: list[int] = []
    for graph_id in range(int(max_graphs)):
        slice_id = float(graph_id % int(max_slices))
        t_base = float((graph_id % 10) * 0.05)

        # ATAR XZ hit
        rows.append(
            [
                0.10 + 0.001 * graph_id,  # x
                0.00,                     # y
                0.20,                     # z
                0.30,                     # E
                t_base,                   # t (normalized)
                1.0,                      # is_xz
                0.0,                      # is_yz
                0.0,                      # is_lyso
                slice_id,                 # slice_id
                t_base * 500.0,           # slice_mean_t (raw ns)
            ]
        )
        batch_ids.append(graph_id)

        # ATAR YZ hit
        rows.append(
            [
                0.00,
                0.12 + 0.001 * graph_id,
                0.25,
                0.25,
                t_base + 0.01,
                0.0,
                1.0,
                0.0,
                slice_id,
                (t_base + 0.01) * 500.0,
            ]
        )
        batch_ids.append(graph_id)

        # LYSO hit
        rows.append(
            [
                0.40,
                0.35,
                0.30,
                0.45,
                t_base + 0.02,
                0.0,
                0.0,
                1.0,
                slice_id,
                (t_base + 0.02) * 500.0,
            ]
        )
        batch_ids.append(graph_id)

        # Second LYSO hit (ensures non-empty LYSO intra-slice edges during trace)
        rows.append(
            [
                0.42,
                0.33,
                0.32,
                0.35,
                t_base + 0.025,
                0.0,
                0.0,
                1.0,
                slice_id,
                (t_base + 0.025) * 500.0,
            ]
        )
        batch_ids.append(graph_id)

    example_x = torch.tensor(rows, dtype=torch.float32, device=device)
    example_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    example_edge_attr = torch.zeros((0, int(edge_dim)), dtype=torch.float32, device=device)
    example_batch = torch.tensor(batch_ids, dtype=torch.long, device=device)
    return example_x, example_edge_index, example_edge_attr, example_batch


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
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        task_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        _ = edge_attr
        if isinstance(data_or_x, Data):
            x_node, node_graph_id = self._resolve_batch_input(data_or_x)
            return self.impl(x_node, node_graph_id, task_weights=task_weights)

        resolved_batch: torch.Tensor | None = batch
        # Support both `(x, batch)` and `(x, edge_index, edge_attr, batch)` call styles.
        if resolved_batch is None and isinstance(edge_index, torch.Tensor) and edge_index.dim() == 1:
            resolved_batch = edge_index
        if resolved_batch is None:
            raise ValueError("PurityModel.forward requires `batch` when called with tensor inputs.")
        return self.impl(data_or_x, resolved_batch, task_weights=task_weights)

    def forward_tensors(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        task_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        # Inference path: if no explicit staged weights are provided, run the
        # full fused task set so event-summary heads are active.
        if task_weights is None and not self.training:
            task_weights = {
                "w_node_pdg": 0.5,
                "w_atar_trigger_slice": 1.0,
                "w_pion_kinematics": 0.0,
                "w_event_builder": 1.0,
                "w_positron_angle": 1.0,
                "w_lyso_condensation": 0.5,
            }
        return self.impl(x, batch, task_weights=task_weights)

    def _architecture_config(self) -> dict[str, Any]:
        return {
            "node_dim": int(self.node_dim),
            "edge_dim": int(self.edge_dim),
            "graph_dim": int(self.graph_dim),
            "hidden": int(self.hidden),
            "heads": int(self.heads),
            "layers": int(self.num_blocks),
            "dropout": float(self.dropout),
            "num_pdg_classes": int(self.num_pdg_classes),
        }

    @staticmethod
    def state_bundle_path_for(model_path: str | Path) -> Path:
        resolved = Path(model_path).expanduser().resolve()
        name = resolved.name
        if name.endswith("_torchscript.pt"):
            prefix = name[: -len("_torchscript.pt")]
            return resolved.with_name(f"{prefix}_state_dict.pt")
        if name.endswith(".pt"):
            return resolved.with_name(f"{resolved.stem}_state_dict.pt")
        return resolved.with_name(f"{name}_state_dict.pt")

    def export_state_bundle(self, path: str | Path) -> Path:
        bundle_path = self.state_bundle_path_for(path)
        payload = {
            "format_version": 1,
            "model_family": "purity",
            "adapter": self.__class__.__name__,
            "architecture": self._architecture_config(),
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
        }
        torch.save(payload, str(bundle_path))
        return bundle_path

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
            example_x, example_edge_index, example_edge_attr, example_batch = _build_trace_example(
                device=device,
                edge_dim=int(self.edge_dim),
            )
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
            output_path = Path(path).expanduser().resolve()
            scripted.save(str(output_path))
            self.export_state_bundle(output_path)
        return scripted

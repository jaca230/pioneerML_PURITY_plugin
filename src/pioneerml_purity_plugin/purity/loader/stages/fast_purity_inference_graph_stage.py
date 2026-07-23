from __future__ import annotations

from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.data_loader.loaders.array_store.ndarray_store import NDArrayStore
from pioneerml.data_loader.loaders.stage.stages.base_stage import BaseStage


class FastPurityInferenceGraphStage(BaseStage):
    """Inference-only PURITY graph builder shared by experimental backends."""

    name = "build_purity_graph"
    requires = ("n_rows",)
    provides = (
        "layout",
        "x_out",
        "edge_index_out",
        "edge_attr_out",
        "graph_event_id",
        "graph_event_id_is_global",
        "graph_time_group_id",
        "node_slice_id_out",
        "slice_graph_id_out",
        "slice_ptr_out",
        "graph_slice_ptr_out",
    )

    def __init__(
        self,
        *,
        input_state_key: str = "features_in",
        node_feature_dim: int = 10,
        edge_feature_dim: int = 11,
        atar_pos_norm: float = 10.0,
        atar_energy_norm: float = 1.0,
        atar_time_norm: float = 500.0,
        lyso_pos_norm: float = 100.0,
        lyso_energy_norm: float = 70.0,
        lyso_time_norm: float = 500.0,
    ) -> None:
        self.input_state_key = input_state_key
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)
        self.norms = np.asarray(
            [
                atar_pos_norm,
                atar_energy_norm,
                atar_time_norm,
                lyso_pos_norm,
                lyso_energy_norm,
                lyso_time_norm,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _list(store: NDArrayStore, field: str, dtype) -> tuple[np.ndarray, np.ndarray]:
        values_key = store.values_key(field)
        offsets_key = store.offsets_key(field, 0)
        if not store.has_raw(values_key) or not store.has_raw(offsets_key):
            return np.empty((0,), dtype=dtype), np.zeros((1,), dtype=np.int64)
        return (
            np.ascontiguousarray(store.values(field), dtype=dtype),
            np.ascontiguousarray(store.offsets(field, 0), dtype=np.int64),
        )

    @staticmethod
    def _aligned_offsets(
        offsets: list[np.ndarray], *, n_rows: int, label: str
    ) -> np.ndarray:
        expected = n_rows + 1
        normalized = []
        for item in offsets:
            if item.shape == (1,) and int(item[0]) == 0:
                item = np.zeros((expected,), dtype=np.int64)
            if item.shape != (expected,):
                raise ValueError(f"{label} offsets have shape {item.shape}; expected {(expected,)}")
            normalized.append(item)
        reference = normalized[0]
        if any(not np.array_equal(reference, item) for item in normalized[1:]):
            raise ValueError(f"{label} columns do not have aligned ragged offsets")
        return np.ascontiguousarray(reference, dtype=np.int64)

    def _inputs(self, store: NDArrayStore, n_rows: int) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        atar_offsets = []
        for field, dtype in (
            ("atar_x", np.float32),
            ("atar_y", np.float32),
            ("atar_z", np.float32),
            ("atar_E", np.float32),
            ("atar_t", np.float32),
            ("atar_view", np.int32),
            ("atar_slice", np.int32),
            ("atar_slice_mean_t", np.float32),
        ):
            values, offsets = self._list(store, field, dtype)
            if field == "atar_slice" and values.size == 0:
                values, offsets = self._list(store, "atar_slice_id", dtype)
            if field in {"atar_slice", "atar_slice_mean_t"} and values.size == 0:
                ref = atar_offsets[0]
                values = np.zeros((int(ref[-1]),), dtype=dtype)
                offsets = ref
            arrays[field] = values
            atar_offsets.append(offsets)

        lyso_offsets = []
        for field, dtype in (
            ("lyso_x", np.float32),
            ("lyso_y", np.float32),
            ("lyso_z", np.float32),
            ("lyso_E", np.float32),
            ("lyso_t", np.float32),
            ("lyso_slice", np.int32),
            ("lyso_slice_mean_t", np.float32),
        ):
            values, offsets = self._list(store, field, dtype)
            if field in {"lyso_slice", "lyso_slice_mean_t"} and values.size == 0:
                ref = lyso_offsets[0]
                values = np.zeros((int(ref[-1]),), dtype=dtype)
                offsets = ref
            arrays[field] = values
            lyso_offsets.append(offsets)

        arrays["atar_offsets"] = self._aligned_offsets(atar_offsets, n_rows=n_rows, label="ATAR")
        arrays["lyso_offsets"] = self._aligned_offsets(lyso_offsets, n_rows=n_rows, label="LYSO")
        return arrays

    @abstractmethod
    def _build_nodes(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
        """Return nodes/counts and optionally native-built slice metadata."""

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        if bool(getattr(owner, "include_targets", False)):
            raise RuntimeError("Experimental fast PURITY graph stages support inference only")
        store = state.get(self.input_state_key)
        if not isinstance(store, NDArrayStore):
            raise RuntimeError(f"{self.name} missing NDArrayStore state '{self.input_state_key}'")

        n_rows = int(state.get("n_rows", 0))
        arrays = self._inputs(store, n_rows)
        native_result = self._build_nodes(arrays)
        nodes, row_node_counts = native_result[:2]
        nodes = np.ascontiguousarray(nodes, dtype=np.float32)
        row_node_counts = np.ascontiguousarray(row_node_counts, dtype=np.int64)

        kept_rows = np.flatnonzero(row_node_counts > 0)
        counts = row_node_counts[kept_rows]
        total_graphs = int(kept_rows.size)
        node_ptr = np.zeros((total_graphs + 1,), dtype=np.int64)
        node_ptr[1:] = np.cumsum(counts, dtype=np.int64)

        event_values = None
        event_key = store.values_key("event_id")
        if store.has_raw(event_key):
            event_values = np.asarray(store.values("event_id"), dtype=np.int64)
        graph_event_id = (
            event_values[kept_rows]
            if event_values is not None
            else kept_rows.astype(np.int64, copy=False)
        )

        if len(native_result) == 6:
            node_slice_id = np.asarray(native_result[2], dtype=np.int64)
            slice_graph_ids = np.asarray(native_result[3], dtype=np.int64)
            slice_counts = np.asarray(native_result[4], dtype=np.int64)
            graph_slice_counts = np.asarray(native_result[5], dtype=np.int64)
        elif nodes.shape[0]:
            input_row_for_node = np.repeat(
                np.arange(n_rows, dtype=np.int64), row_node_counts
            )
            slice_ids = nodes[:, 8].astype(np.int64, copy=False)
            pairs = np.column_stack((input_row_for_node, slice_ids))
            unique_pairs, node_slice_id = np.unique(pairs, axis=0, return_inverse=True)
            input_to_graph = np.full((n_rows,), -1, dtype=np.int64)
            input_to_graph[kept_rows] = np.arange(total_graphs, dtype=np.int64)
            slice_graph_ids = input_to_graph[unique_pairs[:, 0]]
            slice_counts = np.bincount(node_slice_id, minlength=unique_pairs.shape[0])
            graph_slice_counts = np.bincount(
                slice_graph_ids, minlength=total_graphs
            )
        else:
            node_slice_id = np.empty((0,), dtype=np.int64)
            slice_graph_ids = np.empty((0,), dtype=np.int64)
            slice_counts = np.empty((0,), dtype=np.int64)
            graph_slice_counts = np.zeros((total_graphs,), dtype=np.int64)

        slice_ptr = np.zeros((int(slice_counts.size) + 1,), dtype=np.int64)
        slice_ptr[1:] = np.cumsum(slice_counts)
        graph_slice_ptr = np.zeros((total_graphs + 1,), dtype=np.int64)
        graph_slice_ptr[1:] = np.cumsum(graph_slice_counts)
        edge_ptr = np.zeros((total_graphs + 1,), dtype=np.int64)

        state["layout"] = {
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "total_graphs": total_graphs,
        }
        state["x_out"] = nodes
        state["edge_index_out"] = np.zeros((2, 0), dtype=np.int64)
        state["edge_attr_out"] = np.zeros((0, self.edge_feature_dim), dtype=np.float32)
        state["graph_event_id"] = np.ascontiguousarray(graph_event_id, dtype=np.int64)
        state["graph_event_id_is_global"] = event_values is not None
        state["graph_time_group_id"] = np.zeros((total_graphs,), dtype=np.int64)
        state["node_slice_id_out"] = node_slice_id
        state["slice_graph_id_out"] = np.ascontiguousarray(slice_graph_ids, dtype=np.int64)
        state["slice_ptr_out"] = slice_ptr
        state["graph_slice_ptr_out"] = graph_slice_ptr


class OptimizedPythonPurityInferenceGraphStage(FastPurityInferenceGraphStage):
    """Chunk-vectorized NumPy implementation with only event interleaving in Python."""

    def _build_nodes(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        atar_offsets = arrays["atar_offsets"]
        lyso_offsets = arrays["lyso_offsets"]
        atar_counts = np.diff(atar_offsets)
        lyso_counts = np.diff(lyso_offsets)
        row_counts = atar_counts + lyso_counts

        atar = np.zeros((int(atar_offsets[-1]), self.node_feature_dim), dtype=np.float32)
        view = arrays["atar_view"]
        yz = view == 1
        atar[:, 0] = np.where(yz, 0.0, arrays["atar_x"]) / self.norms[0]
        atar[:, 1] = np.where(yz, arrays["atar_y"], 0.0) / self.norms[0]
        atar[:, 2] = arrays["atar_z"] / self.norms[0]
        atar[:, 3] = arrays["atar_E"] / self.norms[1]
        atar[:, 4] = arrays["atar_t"] / self.norms[2]
        atar[:, 5] = view == 0
        atar[:, 6] = yz
        atar[:, 8] = arrays["atar_slice"]
        atar[:, 9] = arrays["atar_slice_mean_t"]

        lyso = np.zeros((int(lyso_offsets[-1]), self.node_feature_dim), dtype=np.float32)
        lyso[:, 0] = arrays["lyso_x"] / self.norms[3]
        lyso[:, 1] = arrays["lyso_y"] / self.norms[3]
        lyso[:, 2] = arrays["lyso_z"] / self.norms[3]
        lyso[:, 3] = arrays["lyso_E"] / self.norms[4]
        lyso[:, 4] = arrays["lyso_t"] / self.norms[5]
        lyso[:, 7] = 1.0
        lyso[:, 8] = arrays["lyso_slice"]
        lyso[:, 9] = arrays["lyso_slice_mean_t"]

        out = np.empty((int(row_counts.sum()), self.node_feature_dim), dtype=np.float32)
        output_offsets = np.concatenate(([0], np.cumsum(row_counts, dtype=np.int64)))
        for row, output_start in enumerate(output_offsets[:-1]):
            na = int(atar_counts[row])
            nl = int(lyso_counts[row])
            ai = int(atar_offsets[row])
            li = int(lyso_offsets[row])
            out[output_start : output_start + na] = atar[ai : ai + na]
            out[output_start + na : output_start + na + nl] = lyso[li : li + nl]
        return out, row_counts


class NativePurityInferenceGraphStage(FastPurityInferenceGraphStage):
    """Thin wrapper around the optional pybind11 chunk builder."""

    def _build_nodes(self, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        try:
            from pioneerml_purity_plugin.purity.loader.native import _purity_loader_native
        except ImportError as exc:
            raise RuntimeError(
                "Native PURITY loader is not built; run loader/native/build_native.py"
            ) from exc
        return tuple(
            np.asarray(item)
            for item in _purity_loader_native.build_graph(arrays, self.norms)
        )

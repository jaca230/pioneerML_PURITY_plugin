from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np

from pioneerml.data_loader.loaders.array_store.ndarray_store import NDArrayStore
from pioneerml.data_loader.loaders.stage.stages.base_stage import BaseStage


class BuildPurityGraphStage(BaseStage):
    """Builds event-level PURITY graph tensors from ragged ATAR/LYSO columns."""

    name = "build_purity_graph"
    MAX_LYSO_OBJECTS = 20
    PDG_PION = 1 << 0
    PDG_MUON = 1 << 1
    PDG_POSITRON = 1 << 2
    PDG_ELECTRON = 1 << 3
    PDG_GAMMA = 1 << 4
    requires = ("n_rows",)
    provides = (
        "layout",
        "x_out",
        "edge_index_out",
        "edge_attr_out",
        "graph_event_id",
        "graph_time_group_id",
        "node_slice_id_out",
        "slice_graph_id_out",
        "slice_ptr_out",
        "graph_slice_ptr_out",
        "y_graph_out",
        "y_event_out",
        "y_slice_out",
        "atar_slice_ptr_out",
        "atar_node_pdg_target_out",
        "atar_true_event_id_out",
        "atar_slice_pdg_target_out",
        "atar_slice_multi_target_out",
        "atar_slice_trigger_target_out",
        "atar_slice_start_target_out",
        "atar_slice_stop_target_out",
        "atar_angle_target_out",
        "atar_pion_stop_target_out",
        "atar_pion_stop_valid_target_out",
        "positron_initial_energy_target_out",
        "lyso_fracs_target_out",
        "lyso_payload_target_out",
        "lyso_mask_target_out",
        "is_trigger_target_out",
        "has_trigger_positron_out",
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
        cache_templates: bool | None = None,
        cache_max_entries: int | None = None,
    ) -> None:
        self.input_state_key = str(input_state_key)
        self.node_feature_dim = int(node_feature_dim)
        self.edge_feature_dim = int(edge_feature_dim)

        self.atar_pos_norm = float(atar_pos_norm)
        self.atar_energy_norm = float(atar_energy_norm)
        self.atar_time_norm = float(atar_time_norm)
        self.lyso_pos_norm = float(lyso_pos_norm)
        self.lyso_energy_norm = float(lyso_energy_norm)
        self.lyso_time_norm = float(lyso_time_norm)

        self.cache_templates = None if cache_templates is None else bool(cache_templates)
        self.cache_max_entries = self._normalize_cache_max_entries(cache_max_entries)
        self._edge_tpl_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _normalize_cache_max_entries(value: int | None) -> int | None:
        if value is None:
            return None
        out = int(value)
        if out <= 0:
            return 0
        return out

    @staticmethod
    def _resolve_effective_cache_templates(*, stage_value: bool | None, owner) -> bool:
        if stage_value is not None:
            return bool(stage_value)
        return bool(getattr(owner, "edge_template_cache_enabled", False))

    @classmethod
    def _resolve_effective_cache_max_entries(cls, *, stage_value: int | None, owner) -> int | None:
        if stage_value is not None:
            return cls._normalize_cache_max_entries(stage_value)
        return cls._normalize_cache_max_entries(getattr(owner, "edge_template_cache_max_entries", None))

    @staticmethod
    def _row_slice(values: np.ndarray, offsets: np.ndarray, row_idx: int) -> np.ndarray:
        start = int(offsets[row_idx])
        stop = int(offsets[row_idx + 1])
        return values[start:stop]

    @staticmethod
    def _scalar_values_or_none(*, store: NDArrayStore, field: str) -> np.ndarray | None:
        key = store.values_key(field)
        if not store.has_raw(key):
            return None
        return np.asarray(store.values(field))

    @staticmethod
    def _list_values_or_empty(*, store: NDArrayStore, field: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
        values_key = store.values_key(field)
        offsets_key = store.offsets_key(field, 0)
        if not store.has_raw(values_key) or not store.has_raw(offsets_key):
            return np.asarray([], dtype=dtype), np.zeros((1,), dtype=np.int64)
        values = np.asarray(store.values(field), dtype=dtype)
        offsets = np.asarray(store.offsets(field, 0), dtype=np.int64)
        return values, offsets

    @staticmethod
    def _zeros_for_offsets(*, ref_offsets: np.ndarray, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
        offsets = np.asarray(ref_offsets, dtype=np.int64)
        if offsets.ndim != 1 or offsets.shape[0] == 0:
            offsets = np.zeros((1,), dtype=np.int64)
        total = int(offsets[-1]) if offsets.shape[0] > 0 else 0
        return np.zeros((max(0, total),), dtype=dtype), offsets

    @staticmethod
    def _list_values_with_aliases(
        *,
        store: NDArrayStore,
        field: str,
        aliases: tuple[str, ...],
        dtype: np.dtype,
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        candidates = (str(field), *tuple(str(a) for a in aliases))
        for candidate in candidates:
            values_key = store.values_key(candidate)
            offsets_key = store.offsets_key(candidate, 0)
            if store.has_raw(values_key) and store.has_raw(offsets_key):
                values = np.asarray(store.values(candidate), dtype=dtype)
                offsets = np.asarray(store.offsets(candidate, 0), dtype=np.int64)
                return values, offsets, candidate
        return np.asarray([], dtype=dtype), np.zeros((1,), dtype=np.int64), None

    def _complete_digraph_cached(
        self,
        count: int,
        *,
        cache_templates: bool,
        cache_max_entries: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if count <= 1:
            empty = np.empty((0,), dtype=np.int64)
            return empty, empty

        if not cache_templates or cache_max_entries == 0:
            src = np.repeat(np.arange(count, dtype=np.int64), count)
            dst = np.tile(np.arange(count, dtype=np.int64), count)
            mask = src != dst
            return src[mask], dst[mask]

        tpl = self._edge_tpl_cache.get(int(count))
        if tpl is not None:
            return tpl

        src = np.repeat(np.arange(count, dtype=np.int64), count)
        dst = np.tile(np.arange(count, dtype=np.int64), count)
        mask = src != dst
        tpl = (src[mask], dst[mask])

        if cache_max_entries is not None and len(self._edge_tpl_cache) >= cache_max_entries:
            oldest = next(iter(self._edge_tpl_cache), None)
            if oldest is not None:
                self._edge_tpl_cache.pop(oldest, None)
        self._edge_tpl_cache[int(count)] = tpl
        return tpl

    @staticmethod
    def _modality_from_nodes(nodes: np.ndarray) -> np.ndarray:
        is_lyso = nodes[:, 7] > 0.5
        is_yz = nodes[:, 6] > 0.5
        return np.where(is_lyso, 2, np.where(is_yz, 1, 0)).astype(np.int64, copy=False)

    @classmethod
    def _build_purity_edge_attr(cls, nodes: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        edge_count = int(src.shape[0])
        if edge_count == 0:
            return np.zeros((0, 11), dtype=np.float32)

        u = nodes[src]
        v = nodes[dst]

        u_mod = cls._modality_from_nodes(u)
        v_mod = cls._modality_from_nodes(v)

        u_scale = np.ones((edge_count, 5), dtype=np.float32)
        v_scale = np.ones((edge_count, 5), dtype=np.float32)

        u_atar = u_mod < 2
        v_atar = v_mod < 2

        u_scale[:, :3] = np.where(u_atar[:, None], 10.0, 100.0)
        v_scale[:, :3] = np.where(v_atar[:, None], 10.0, 100.0)
        u_scale[:, 3] = np.where(u_atar, 1.0, 70.0)
        v_scale[:, 3] = np.where(v_atar, 1.0, 70.0)
        u_scale[:, 4] = 500.0
        v_scale[:, 4] = 500.0

        physical_u = u[:, :5] * u_scale
        physical_v = v[:, :5] * v_scale
        diffs = physical_v - physical_u

        is_pure_atar = u_atar & v_atar
        diffs[:, :3] /= np.where(is_pure_atar[:, None], 10.0, 100.0)
        diffs[:, 3] /= np.where(is_pure_atar, 1.0, 70.0)
        diffs[:, 4] /= 500.0

        m_xz_xz = (u_mod == 0) & (v_mod == 0)
        m_yz_yz = (u_mod == 1) & (v_mod == 1)
        m_xz_yz = ((u_mod == 0) & (v_mod == 1)) | ((u_mod == 1) & (v_mod == 0))
        m_calo_calo = (u_mod == 2) & (v_mod == 2)
        m_xz_calo = ((u_mod == 0) & (v_mod == 2)) | ((u_mod == 2) & (v_mod == 0))
        m_yz_calo = ((u_mod == 1) & (v_mod == 2)) | ((u_mod == 2) & (v_mod == 1))

        out = np.zeros((edge_count, 11), dtype=np.float32)
        out[:, :5] = diffs.astype(np.float32, copy=False)

        out[m_xz_xz, 1] = 0.0
        out[m_xz_xz, 5] = 1.0

        out[m_yz_yz, 0] = 0.0
        out[m_yz_yz, 6] = 1.0

        out[m_xz_yz, 0] = 0.0
        out[m_xz_yz, 1] = 0.0
        out[m_xz_yz, 7] = 1.0

        out[m_calo_calo, 8] = 1.0

        out[m_xz_calo, 3] = 0.0
        out[m_xz_calo, 9] = 1.0

        out[m_yz_calo, 3] = 0.0
        out[m_yz_calo, 10] = 1.0
        return out

    def _build_nodes_for_row(
        self,
        *,
        row_idx: int,
        atar_x_vals: np.ndarray,
        atar_x_off: np.ndarray,
        atar_y_vals: np.ndarray,
        atar_y_off: np.ndarray,
        atar_z_vals: np.ndarray,
        atar_z_off: np.ndarray,
        atar_e_vals: np.ndarray,
        atar_e_off: np.ndarray,
        atar_t_vals: np.ndarray,
        atar_t_off: np.ndarray,
        atar_view_vals: np.ndarray,
        atar_view_off: np.ndarray,
        atar_slice_vals: np.ndarray,
        atar_slice_off: np.ndarray,
        atar_slice_mean_t_vals: np.ndarray,
        atar_slice_mean_t_off: np.ndarray,
        lyso_x_vals: np.ndarray,
        lyso_x_off: np.ndarray,
        lyso_y_vals: np.ndarray,
        lyso_y_off: np.ndarray,
        lyso_z_vals: np.ndarray,
        lyso_z_off: np.ndarray,
        lyso_e_vals: np.ndarray,
        lyso_e_off: np.ndarray,
        lyso_t_vals: np.ndarray,
        lyso_t_off: np.ndarray,
        lyso_slice_vals: np.ndarray,
        lyso_slice_off: np.ndarray,
        lyso_slice_mean_t_vals: np.ndarray,
        lyso_slice_mean_t_off: np.ndarray,
    ) -> np.ndarray:
        atar_x = self._row_slice(atar_x_vals, atar_x_off, row_idx)
        atar_y = self._row_slice(atar_y_vals, atar_y_off, row_idx)
        atar_z = self._row_slice(atar_z_vals, atar_z_off, row_idx)
        atar_e = self._row_slice(atar_e_vals, atar_e_off, row_idx)
        atar_t = self._row_slice(atar_t_vals, atar_t_off, row_idx)
        atar_view = self._row_slice(atar_view_vals, atar_view_off, row_idx)
        atar_slice = self._row_slice(atar_slice_vals, atar_slice_off, row_idx)
        atar_slice_mean_t = self._row_slice(atar_slice_mean_t_vals, atar_slice_mean_t_off, row_idx)

        n_atar = min(
            int(atar_x.shape[0]),
            int(atar_y.shape[0]),
            int(atar_z.shape[0]),
            int(atar_e.shape[0]),
            int(atar_t.shape[0]),
            int(atar_view.shape[0]),
            int(atar_slice.shape[0]),
            int(atar_slice_mean_t.shape[0]),
        )

        lyso_x = self._row_slice(lyso_x_vals, lyso_x_off, row_idx)
        lyso_y = self._row_slice(lyso_y_vals, lyso_y_off, row_idx)
        lyso_z = self._row_slice(lyso_z_vals, lyso_z_off, row_idx)
        lyso_e = self._row_slice(lyso_e_vals, lyso_e_off, row_idx)
        lyso_t = self._row_slice(lyso_t_vals, lyso_t_off, row_idx)
        lyso_slice = self._row_slice(lyso_slice_vals, lyso_slice_off, row_idx)
        lyso_slice_mean_t = self._row_slice(lyso_slice_mean_t_vals, lyso_slice_mean_t_off, row_idx)

        n_lyso = min(
            int(lyso_x.shape[0]),
            int(lyso_y.shape[0]),
            int(lyso_z.shape[0]),
            int(lyso_e.shape[0]),
            int(lyso_t.shape[0]),
            int(lyso_slice.shape[0]),
            int(lyso_slice_mean_t.shape[0]),
        )

        if n_atar <= 0 and n_lyso <= 0:
            return np.zeros((0, self.node_feature_dim), dtype=np.float32)

        atar_nodes = np.zeros((max(0, n_atar), self.node_feature_dim), dtype=np.float32)
        if n_atar > 0:
            atar_view = atar_view[:n_atar].astype(np.int32, copy=False)
            is_yz = atar_view == 1
            # Strict Omar parity (deprecated/unified_reco/dataset.py):
            # - XZ hits: keep x, zero y
            # - YZ hits: keep y, zero x
            # This avoids leaking both coordinates into single-view ATAR hits.
            atar_x_view = np.where(is_yz, 0.0, atar_x[:n_atar]).astype(np.float32, copy=False)
            atar_y_view = np.where(is_yz, atar_y[:n_atar], 0.0).astype(np.float32, copy=False)
            atar_nodes[:, 0] = atar_x_view / self.atar_pos_norm
            atar_nodes[:, 1] = atar_y_view / self.atar_pos_norm
            atar_nodes[:, 2] = atar_z[:n_atar].astype(np.float32, copy=False) / self.atar_pos_norm
            atar_nodes[:, 3] = atar_e[:n_atar].astype(np.float32, copy=False) / self.atar_energy_norm
            atar_nodes[:, 4] = atar_t[:n_atar].astype(np.float32, copy=False) / self.atar_time_norm
            atar_nodes[:, 5] = (atar_view == 0).astype(np.float32, copy=False)
            atar_nodes[:, 6] = (atar_view == 1).astype(np.float32, copy=False)
            atar_nodes[:, 7] = 0.0
            atar_nodes[:, 8] = atar_slice[:n_atar].astype(np.float32, copy=False)
            # Omar parity: column 9 stores raw per-hit slice mean time (ns).
            atar_nodes[:, 9] = atar_slice_mean_t[:n_atar].astype(np.float32, copy=False)

        lyso_nodes = np.zeros((max(0, n_lyso), self.node_feature_dim), dtype=np.float32)
        if n_lyso > 0:
            lyso_nodes[:, 0] = lyso_x[:n_lyso].astype(np.float32, copy=False) / self.lyso_pos_norm
            lyso_nodes[:, 1] = lyso_y[:n_lyso].astype(np.float32, copy=False) / self.lyso_pos_norm
            lyso_nodes[:, 2] = lyso_z[:n_lyso].astype(np.float32, copy=False) / self.lyso_pos_norm
            lyso_nodes[:, 3] = lyso_e[:n_lyso].astype(np.float32, copy=False) / self.lyso_energy_norm
            lyso_nodes[:, 4] = lyso_t[:n_lyso].astype(np.float32, copy=False) / self.lyso_time_norm
            lyso_nodes[:, 5] = 0.0
            lyso_nodes[:, 6] = 0.0
            lyso_nodes[:, 7] = 1.0
            lyso_nodes[:, 8] = lyso_slice[:n_lyso].astype(np.float32, copy=False)
            # Omar parity: column 9 stores raw per-hit slice mean time (ns).
            lyso_nodes[:, 9] = lyso_slice_mean_t[:n_lyso].astype(np.float32, copy=False)

        if n_atar <= 0:
            return lyso_nodes
        if n_lyso <= 0:
            return atar_nodes
        return np.concatenate([atar_nodes, lyso_nodes], axis=0)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        chunk_in = state.get(self.input_state_key)
        if not isinstance(chunk_in, NDArrayStore):
            raise RuntimeError(f"{self.name} missing NDArrayStore state '{self.input_state_key}'.")

        n_rows = int(state.get("n_rows", 0))
        if n_rows <= 0:
            state["layout"] = {
                "node_ptr": np.zeros((1,), dtype=np.int64),
                "edge_ptr": np.zeros((1,), dtype=np.int64),
                "total_graphs": 0,
            }
            state["x_out"] = np.zeros((0, self.node_feature_dim), dtype=np.float32)
            state["edge_attr_out"] = np.zeros((0, self.edge_feature_dim), dtype=np.float32)
            state["edge_index_out"] = np.zeros((2, 0), dtype=np.int64)
            state["graph_event_id"] = np.zeros((0,), dtype=np.int64)
            state["graph_time_group_id"] = np.zeros((0,), dtype=np.int64)
            state["node_slice_id_out"] = np.zeros((0,), dtype=np.int64)
            state["slice_graph_id_out"] = np.zeros((0,), dtype=np.int64)
            state["slice_ptr_out"] = np.zeros((1,), dtype=np.int64)
            state["graph_slice_ptr_out"] = np.zeros((1,), dtype=np.int64)
            state["atar_slice_ptr_out"] = np.zeros((1,), dtype=np.int64)
            state["y_graph_out"] = np.zeros((0, 1), dtype=np.float32)
            state["y_event_out"] = np.zeros((0, 1), dtype=np.float32)
            state["y_slice_out"] = np.zeros((0, 1), dtype=np.float32)
            state["atar_node_pdg_target_out"] = np.zeros((0, 3), dtype=np.float32)
            state["atar_true_event_id_out"] = np.zeros((0,), dtype=np.int64)
            state["atar_slice_pdg_target_out"] = np.zeros((0, 3), dtype=np.float32)
            state["atar_slice_multi_target_out"] = np.zeros((0,), dtype=np.float32)
            state["atar_slice_trigger_target_out"] = np.zeros((0,), dtype=np.float32)
            state["atar_slice_start_target_out"] = np.zeros((0, 3), dtype=np.float32)
            state["atar_slice_stop_target_out"] = np.zeros((0, 3), dtype=np.float32)
            state["atar_angle_target_out"] = np.zeros((0, 3), dtype=np.float32)
            state["atar_pion_stop_target_out"] = np.zeros((0, 3), dtype=np.float32)
            state["positron_initial_energy_target_out"] = np.zeros((0, 1), dtype=np.float32)
            state["lyso_fracs_target_out"] = np.zeros((0, self.MAX_LYSO_OBJECTS), dtype=np.float32)
            state["lyso_payload_target_out"] = np.zeros((0, self.MAX_LYSO_OBJECTS, 4), dtype=np.float32)
            state["lyso_mask_target_out"] = np.zeros((0, self.MAX_LYSO_OBJECTS), dtype=np.float32)
            state["is_trigger_target_out"] = np.zeros((0,), dtype=np.float32)
            state["has_trigger_positron_out"] = np.zeros((0,), dtype=np.float32)
            return

        def ensure_row_offsets(offsets: np.ndarray) -> np.ndarray:
            """
            Ensure list offsets are compatible with row-wise slicing.

            We occasionally receive optional list columns as missing, which maps to
            offsets shape (1,) from _list_values_or_empty. For per-row slicing we
            need shape (n_rows + 1), so synthesize an all-zero offsets vector.
            """
            off = np.asarray(offsets, dtype=np.int64)
            expected = int(n_rows) + 1
            if expected <= 1:
                return np.zeros((1,), dtype=np.int64)
            if int(off.ndim) != 1 or int(off.shape[0]) == 0:
                return np.zeros((expected,), dtype=np.int64)
            if int(off.shape[0]) == expected:
                return off
            if int(off.shape[0]) == 1:
                return np.zeros((expected,), dtype=np.int64)
            if int(off.shape[0]) > expected:
                return off[:expected]
            pad = np.full((expected - int(off.shape[0]),), int(off[-1]), dtype=np.int64)
            return np.concatenate([off, pad], axis=0)

        atar_x_vals, atar_x_off = self._list_values_or_empty(store=chunk_in, field="atar_x", dtype=np.float32)
        atar_y_vals, atar_y_off = self._list_values_or_empty(store=chunk_in, field="atar_y", dtype=np.float32)
        atar_z_vals, atar_z_off = self._list_values_or_empty(store=chunk_in, field="atar_z", dtype=np.float32)
        atar_e_vals, atar_e_off = self._list_values_or_empty(store=chunk_in, field="atar_E", dtype=np.float32)
        atar_t_vals, atar_t_off = self._list_values_or_empty(store=chunk_in, field="atar_t", dtype=np.float32)
        atar_view_vals, atar_view_off = self._list_values_or_empty(store=chunk_in, field="atar_view", dtype=np.int32)
        atar_slice_vals, atar_slice_off, atar_slice_field = self._list_values_with_aliases(
            store=chunk_in,
            field="atar_slice",
            aliases=("atar_slice_id",),
            dtype=np.int32,
        )
        if atar_slice_field is None:
            atar_slice_vals, atar_slice_off = self._zeros_for_offsets(ref_offsets=atar_t_off, dtype=np.int32)
        atar_slice_mean_t_vals, atar_slice_mean_t_off = self._list_values_or_empty(
            store=chunk_in,
            field="atar_slice_mean_t",
            dtype=np.float32,
        )
        if int(atar_slice_mean_t_off[-1]) == 0 and int(atar_t_off[-1]) > 0:
            atar_slice_mean_t_vals, atar_slice_mean_t_off = self._zeros_for_offsets(
                ref_offsets=atar_t_off,
                dtype=np.float32,
            )

        lyso_x_vals, lyso_x_off = self._list_values_or_empty(store=chunk_in, field="lyso_x", dtype=np.float32)
        lyso_y_vals, lyso_y_off = self._list_values_or_empty(store=chunk_in, field="lyso_y", dtype=np.float32)
        lyso_z_vals, lyso_z_off = self._list_values_or_empty(store=chunk_in, field="lyso_z", dtype=np.float32)
        lyso_e_vals, lyso_e_off = self._list_values_or_empty(store=chunk_in, field="lyso_E", dtype=np.float32)
        lyso_t_vals, lyso_t_off = self._list_values_or_empty(store=chunk_in, field="lyso_t", dtype=np.float32)
        lyso_slice_vals, lyso_slice_off = self._list_values_or_empty(store=chunk_in, field="lyso_slice", dtype=np.int32)
        if int(lyso_slice_off[-1]) == 0 and int(lyso_t_off[-1]) > 0:
            lyso_slice_vals, lyso_slice_off = self._zeros_for_offsets(ref_offsets=lyso_t_off, dtype=np.int32)
        lyso_slice_mean_t_vals, lyso_slice_mean_t_off = self._list_values_or_empty(
            store=chunk_in,
            field="lyso_slice_mean_t",
            dtype=np.float32,
        )
        if int(lyso_slice_mean_t_off[-1]) == 0 and int(lyso_t_off[-1]) > 0:
            lyso_slice_mean_t_vals, lyso_slice_mean_t_off = self._zeros_for_offsets(
                ref_offsets=lyso_t_off,
                dtype=np.float32,
            )

        event_id_values = self._scalar_values_or_none(store=chunk_in, field="event_id")
        target_signal = self._scalar_values_or_none(store=chunk_in, field="truth_is_signal")
        target_energy = self._scalar_values_or_none(store=chunk_in, field="truth_positron_energy")
        truth_theta = self._scalar_values_or_none(store=chunk_in, field="truth_theta")
        truth_phi = self._scalar_values_or_none(store=chunk_in, field="truth_phi")
        truth_pion_stop_x = self._scalar_values_or_none(store=chunk_in, field="truth_pion_stop_x")
        truth_pion_stop_y = self._scalar_values_or_none(store=chunk_in, field="truth_pion_stop_y")
        truth_pion_stop_z = self._scalar_values_or_none(store=chunk_in, field="truth_pion_stop_z")

        atar_pdg_vals, atar_pdg_off = self._list_values_or_empty(store=chunk_in, field="atar_pdg", dtype=np.int64)
        if int(atar_pdg_off[-1]) == 0 and int(atar_x_off[-1]) > 0:
            atar_pdg_vals, atar_pdg_off = self._zeros_for_offsets(ref_offsets=atar_x_off, dtype=np.int64)
        atar_origin_vals, atar_origin_off = self._list_values_or_empty(store=chunk_in, field="atar_origin", dtype=np.int64)
        if int(atar_origin_off[-1]) == 0 and int(atar_x_off[-1]) > 0:
            atar_origin_vals, atar_origin_off = self._zeros_for_offsets(ref_offsets=atar_x_off, dtype=np.int64)
        atar_truth_t_vals, atar_truth_t_off = self._list_values_or_empty(store=chunk_in, field="atar_truth_t", dtype=np.float32)
        if int(atar_truth_t_off[-1]) == 0 and int(atar_x_off[-1]) > 0:
            atar_truth_t_vals, atar_truth_t_off = self._zeros_for_offsets(ref_offsets=atar_x_off, dtype=np.float32)
        lyso_origin_vals, lyso_origin_off = self._list_values_or_empty(store=chunk_in, field="lyso_origin", dtype=np.int64)
        if int(lyso_origin_off[-1]) == 0 and int(lyso_x_off[-1]) > 0:
            lyso_origin_vals, lyso_origin_off = self._zeros_for_offsets(ref_offsets=lyso_x_off, dtype=np.int64)

        # Normalize all list offsets to n_rows+1 so _row_slice is safe for optional
        # columns that are missing and represented with shape (1,) offsets.
        atar_x_off = ensure_row_offsets(atar_x_off)
        atar_y_off = ensure_row_offsets(atar_y_off)
        atar_z_off = ensure_row_offsets(atar_z_off)
        atar_e_off = ensure_row_offsets(atar_e_off)
        atar_t_off = ensure_row_offsets(atar_t_off)
        atar_view_off = ensure_row_offsets(atar_view_off)
        atar_slice_off = ensure_row_offsets(atar_slice_off)
        atar_slice_mean_t_off = ensure_row_offsets(atar_slice_mean_t_off)
        lyso_x_off = ensure_row_offsets(lyso_x_off)
        lyso_y_off = ensure_row_offsets(lyso_y_off)
        lyso_z_off = ensure_row_offsets(lyso_z_off)
        lyso_e_off = ensure_row_offsets(lyso_e_off)
        lyso_t_off = ensure_row_offsets(lyso_t_off)
        lyso_slice_off = ensure_row_offsets(lyso_slice_off)
        lyso_slice_mean_t_off = ensure_row_offsets(lyso_slice_mean_t_off)
        atar_pdg_off = ensure_row_offsets(atar_pdg_off)
        atar_origin_off = ensure_row_offsets(atar_origin_off)
        atar_truth_t_off = ensure_row_offsets(atar_truth_t_off)
        lyso_origin_off = ensure_row_offsets(lyso_origin_off)

        include_targets = bool(getattr(owner, "include_targets", False))
        if include_targets and target_signal is None and target_energy is None:
            raise RuntimeError(
                "PURITY training/eval requires either 'truth_is_signal' or 'truth_positron_energy' target columns."
            )

        def fit_length(arr: np.ndarray, *, length: int, fill_value: float | int = 0.0, dtype: np.dtype) -> np.ndarray:
            out = np.full((max(0, int(length)),), fill_value, dtype=dtype)
            if int(length) <= 0:
                return out
            n_copy = min(int(length), int(arr.shape[0]))
            if n_copy > 0:
                out[:n_copy] = arr[:n_copy].astype(dtype, copy=False)
            return out

        cache_templates = self._resolve_effective_cache_templates(stage_value=self.cache_templates, owner=owner)
        cache_max_entries = self._resolve_effective_cache_max_entries(
            stage_value=self.cache_max_entries,
            owner=owner,
        )

        node_blocks: list[np.ndarray] = []
        edge_attr_blocks: list[np.ndarray] = []
        edge_index_blocks: list[np.ndarray] = []
        graph_event_ids: list[int] = []
        graph_targets: list[float] = []
        slice_targets: list[float] = []
        node_counts: list[int] = []
        edge_counts: list[int] = []
        graph_slice_counts: list[int] = []
        graph_atar_slice_counts: list[int] = []
        slice_node_counts: list[int] = []
        slice_graph_ids: list[int] = []
        node_slice_id_blocks: list[np.ndarray] = []
        atar_node_pdg_target_blocks: list[np.ndarray] = []
        atar_true_event_id_blocks: list[np.ndarray] = []
        lyso_fracs_target_blocks: list[np.ndarray] = []
        is_trigger_target_blocks: list[np.ndarray] = []
        atar_slice_pdg_target_rows: list[np.ndarray] = []
        atar_slice_multi_target_rows: list[np.ndarray] = []
        atar_slice_trigger_target_rows: list[np.ndarray] = []
        atar_slice_start_target_rows: list[np.ndarray] = []
        atar_slice_stop_target_rows: list[np.ndarray] = []
        atar_angle_target_rows: list[np.ndarray] = []
        atar_pion_stop_target_rows: list[np.ndarray] = []
        atar_pion_stop_valid_target_rows: list[np.ndarray] = []
        lyso_payload_target_rows: list[np.ndarray] = []
        lyso_mask_target_rows: list[np.ndarray] = []
        positron_energy_target_rows: list[np.ndarray] = []
        has_trigger_positron_rows: list[float] = []
        node_base = 0
        slice_base = 0

        for row_idx in range(n_rows):
            nodes = self._build_nodes_for_row(
                row_idx=row_idx,
                atar_x_vals=atar_x_vals,
                atar_x_off=atar_x_off,
                atar_y_vals=atar_y_vals,
                atar_y_off=atar_y_off,
                atar_z_vals=atar_z_vals,
                atar_z_off=atar_z_off,
                atar_e_vals=atar_e_vals,
                atar_e_off=atar_e_off,
                atar_t_vals=atar_t_vals,
                atar_t_off=atar_t_off,
                atar_view_vals=atar_view_vals,
                atar_view_off=atar_view_off,
                atar_slice_vals=atar_slice_vals,
                atar_slice_off=atar_slice_off,
                atar_slice_mean_t_vals=atar_slice_mean_t_vals,
                atar_slice_mean_t_off=atar_slice_mean_t_off,
                lyso_x_vals=lyso_x_vals,
                lyso_x_off=lyso_x_off,
                lyso_y_vals=lyso_y_vals,
                lyso_y_off=lyso_y_off,
                lyso_z_vals=lyso_z_vals,
                lyso_z_off=lyso_z_off,
                lyso_e_vals=lyso_e_vals,
                lyso_e_off=lyso_e_off,
                lyso_t_vals=lyso_t_vals,
                lyso_t_off=lyso_t_off,
                lyso_slice_vals=lyso_slice_vals,
                lyso_slice_off=lyso_slice_off,
                lyso_slice_mean_t_vals=lyso_slice_mean_t_vals,
                lyso_slice_mean_t_off=lyso_slice_mean_t_off,
            )
            n_nodes = int(nodes.shape[0])
            if n_nodes <= 0:
                continue

            src_local, dst_local = self._complete_digraph_cached(
                n_nodes,
                cache_templates=cache_templates,
                cache_max_entries=cache_max_entries,
            )
            edge_attr = self._build_purity_edge_attr(nodes, src_local, dst_local)

            if src_local.size > 0:
                edge_index = np.vstack([src_local + node_base, dst_local + node_base]).astype(np.int64, copy=False)
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)

            node_blocks.append(nodes)
            edge_attr_blocks.append(edge_attr)
            edge_index_blocks.append(edge_index)
            graph_idx = int(len(node_counts))
            if event_id_values is not None and int(event_id_values.shape[0]) > int(row_idx):
                graph_event_ids.append(int(event_id_values[row_idx]))
            else:
                graph_event_ids.append(int(row_idx))
            node_counts.append(int(n_nodes))
            edge_counts.append(int(edge_attr.shape[0]))

            node_slice_ids = nodes[:, 8].astype(np.int64, copy=False)
            if int(node_slice_ids.size) > 0:
                _, inv = np.unique(node_slice_ids, return_inverse=True)
                local_slice_ids = inv.astype(np.int64, copy=False)
                n_slices = int(local_slice_ids.max() + 1) if int(local_slice_ids.size) > 0 else 0
                node_slice_id_blocks.append(local_slice_ids + int(slice_base))
                if n_slices > 0:
                    slice_counts = np.bincount(local_slice_ids, minlength=n_slices).astype(np.int64, copy=False)
                    slice_node_counts.extend(int(v) for v in slice_counts.tolist())
                    slice_graph_ids.extend([graph_idx] * n_slices)
                    graph_slice_counts.append(n_slices)
                    if include_targets:
                        if target_signal is not None:
                            target_value = float(target_signal[row_idx])
                        elif target_energy is not None:
                            target_value = 1.0 if float(target_energy[row_idx]) > 0.0 else 0.0
                        else:
                            target_value = 0.0
                        slice_targets.extend([target_value] * n_slices)
                    slice_base += n_slices
                else:
                    graph_slice_counts.append(0)
            else:
                graph_slice_counts.append(0)

            n_atar = int(np.sum(nodes[:, 7] < 0.5))
            n_lyso = int(max(0, n_nodes - n_atar))

            atar_pdg_row = fit_length(
                self._row_slice(atar_pdg_vals, atar_pdg_off, row_idx),
                length=n_atar,
                fill_value=0,
                dtype=np.int64,
            )
            atar_origin_row = fit_length(
                self._row_slice(atar_origin_vals, atar_origin_off, row_idx),
                length=n_atar,
                fill_value=0,
                dtype=np.int64,
            )
            atar_truth_t_row = fit_length(
                self._row_slice(atar_truth_t_vals, atar_truth_t_off, row_idx),
                length=n_atar,
                fill_value=0.0,
                dtype=np.float32,
            )
            lyso_origin_row = fit_length(
                self._row_slice(lyso_origin_vals, lyso_origin_off, row_idx),
                length=n_lyso,
                fill_value=-1,
                dtype=np.int64,
            )

            atar_node_pdg_target = np.zeros((n_nodes, 3), dtype=np.float32)
            if n_atar > 0:
                is_pion = (atar_pdg_row & self.PDG_PION) > 0
                is_muon = (atar_pdg_row & self.PDG_MUON) > 0
                is_mip = ((atar_pdg_row & self.PDG_POSITRON) > 0) | ((atar_pdg_row & self.PDG_ELECTRON) > 0) | (
                    (atar_pdg_row & self.PDG_GAMMA) > 0
                )
                atar_node_pdg_target[:n_atar, 0] = is_pion.astype(np.float32, copy=False)
                atar_node_pdg_target[:n_atar, 1] = is_muon.astype(np.float32, copy=False)
                atar_node_pdg_target[:n_atar, 2] = is_mip.astype(np.float32, copy=False)
            atar_node_pdg_target_blocks.append(atar_node_pdg_target)

            atar_true_event_id = np.full((n_nodes,), -1, dtype=np.int64)
            if n_atar > 0:
                atar_true_event_id[:n_atar] = atar_origin_row
            atar_true_event_id_blocks.append(atar_true_event_id)

            is_trigger = np.zeros((n_nodes,), dtype=np.float32)
            if n_atar > 0:
                is_trigger[:n_atar] = (atar_origin_row == 0).astype(np.float32, copy=False)
            if n_lyso > 0:
                is_trigger[n_atar:] = (lyso_origin_row == 0).astype(np.float32, copy=False)
            is_trigger_target_blocks.append(is_trigger)

            lyso_fracs = np.zeros((n_nodes, self.MAX_LYSO_OBJECTS), dtype=np.float32)
            lyso_payload = np.zeros((self.MAX_LYSO_OBJECTS, 4), dtype=np.float32)
            lyso_mask = np.zeros((self.MAX_LYSO_OBJECTS,), dtype=np.float32)
            if n_lyso > 0:
                lyso_x_row = fit_length(
                    self._row_slice(lyso_x_vals, lyso_x_off, row_idx),
                    length=n_lyso,
                    fill_value=0.0,
                    dtype=np.float32,
                )
                lyso_y_row = fit_length(
                    self._row_slice(lyso_y_vals, lyso_y_off, row_idx),
                    length=n_lyso,
                    fill_value=0.0,
                    dtype=np.float32,
                )
                lyso_z_row = fit_length(
                    self._row_slice(lyso_z_vals, lyso_z_off, row_idx),
                    length=n_lyso,
                    fill_value=0.0,
                    dtype=np.float32,
                )
                lyso_e_row = fit_length(
                    self._row_slice(lyso_e_vals, lyso_e_off, row_idx),
                    length=n_lyso,
                    fill_value=0.0,
                    dtype=np.float32,
                )
                unique_objs = np.unique(lyso_origin_row)
                unique_objs = unique_objs[unique_objs >= 0]
                n_objs_true = min(int(unique_objs.shape[0]), int(self.MAX_LYSO_OBJECTS))
                for obj_idx in range(n_objs_true):
                    obj_id = unique_objs[obj_idx]
                    obj_mask = lyso_origin_row == obj_id
                    lyso_fracs[n_atar:, obj_idx] = obj_mask.astype(np.float32, copy=False)
                    lyso_mask[obj_idx] = 1.0
                    e_sum = float(np.sum(lyso_e_row[obj_mask]))
                    if e_sum > 0.0:
                        cx = float(np.sum(lyso_x_row[obj_mask] * lyso_e_row[obj_mask]) / e_sum)
                        cy = float(np.sum(lyso_y_row[obj_mask] * lyso_e_row[obj_mask]) / e_sum)
                        cz = float(np.sum(lyso_z_row[obj_mask] * lyso_e_row[obj_mask]) / e_sum)
                        lyso_payload[obj_idx, 0] = cx / self.lyso_pos_norm
                        lyso_payload[obj_idx, 1] = cy / self.lyso_pos_norm
                        lyso_payload[obj_idx, 2] = cz / self.lyso_pos_norm
                        lyso_payload[obj_idx, 3] = e_sum / self.lyso_energy_norm
            lyso_fracs_target_blocks.append(lyso_fracs)
            lyso_payload_target_rows.append(lyso_payload)
            lyso_mask_target_rows.append(lyso_mask)

            has_trigger_positron = 0.0
            if n_atar > 0:
                has_trigger_positron = float(
                    np.any((atar_origin_row == 0) & ((atar_pdg_row & self.PDG_POSITRON) > 0))
                )
            has_trigger_positron_rows.append(has_trigger_positron)

            if target_energy is not None and int(target_energy.shape[0]) > int(row_idx):
                positron_energy_target_rows.append(np.asarray([float(target_energy[row_idx])], dtype=np.float32))
            else:
                positron_energy_target_rows.append(np.asarray([0.0], dtype=np.float32))

            n_atar_slices = 0
            if n_atar > 0:
                atar_slice_ids = nodes[:n_atar, 8].astype(np.int64, copy=False)
                unique_atar_slices, atar_slice_inverse = np.unique(atar_slice_ids, return_inverse=True)
                n_atar_slices = int(unique_atar_slices.shape[0])

                atar_x_row = fit_length(
                    self._row_slice(atar_x_vals, atar_x_off, row_idx),
                    length=n_atar,
                    fill_value=0.0,
                    dtype=np.float32,
                )
                atar_y_row = fit_length(
                    self._row_slice(atar_y_vals, atar_y_off, row_idx),
                    length=n_atar,
                    fill_value=0.0,
                    dtype=np.float32,
                )
                atar_z_row = fit_length(
                    self._row_slice(atar_z_vals, atar_z_off, row_idx),
                    length=n_atar,
                    fill_value=0.0,
                    dtype=np.float32,
                )

                for local_slice_idx in range(n_atar_slices):
                    s_mask = atar_slice_inverse == local_slice_idx
                    s_origins = atar_origin_row[s_mask]
                    s_pdgs = atar_pdg_row[s_mask]
                    s_idx = np.where(s_mask)[0]

                    multi_event = 0.0
                    if int(s_origins.shape[0]) > 1 and np.any(s_origins != s_origins[0]):
                        multi_event = 1.0
                    atar_slice_multi_target_rows.append(np.asarray([multi_event], dtype=np.float32))
                    atar_slice_trigger_target_rows.append(
                        np.asarray([float(np.any(s_origins == 0))], dtype=np.float32)
                    )

                    trigger_mask = s_origins == 0
                    if bool(np.any(trigger_mask)):
                        s_pion = (s_pdgs & self.PDG_PION) > 0
                        s_muon = (s_pdgs & self.PDG_MUON) > 0
                        s_posi = (s_pdgs & self.PDG_POSITRON) > 0
                        if bool(np.any(trigger_mask & s_pion)):
                            ref_mask = trigger_mask & s_pion
                        elif bool(np.any(trigger_mask & s_muon)):
                            ref_mask = trigger_mask & s_muon
                        elif bool(np.any(trigger_mask & s_posi)):
                            ref_mask = trigger_mask & s_posi
                        else:
                            ref_mask = trigger_mask
                    else:
                        ref_mask = np.ones_like(s_mask[s_mask], dtype=bool)

                    ref_idx = s_idx[ref_mask]
                    s_pdg_out = np.zeros((3,), dtype=np.float32)
                    if int(ref_idx.shape[0]) > 0:
                        s_pdg_out[0] = float(np.any((atar_pdg_row[ref_idx] & self.PDG_PION) > 0))
                        s_pdg_out[1] = float(np.any((atar_pdg_row[ref_idx] & self.PDG_MUON) > 0))
                        s_pdg_out[2] = float(
                            np.any(
                                ((atar_pdg_row[ref_idx] & self.PDG_POSITRON) > 0)
                                | ((atar_pdg_row[ref_idx] & self.PDG_ELECTRON) > 0)
                                | ((atar_pdg_row[ref_idx] & self.PDG_GAMMA) > 0)
                            )
                        )
                    atar_slice_pdg_target_rows.append(s_pdg_out.reshape(1, 3))

                    if int(ref_idx.shape[0]) > 0:
                        s_t = atar_truth_t_row[ref_idx]
                        s_x = atar_x_row[ref_idx]
                        s_y = atar_y_row[ref_idx]
                        s_z = atar_z_row[ref_idx]
                        t_min_idx = int(np.argmin(s_t))
                        t_max_idx = int(np.argmax(s_t))
                        if float(s_z[t_min_idx]) < float(s_z[t_max_idx]):
                            low_idx, high_idx = t_min_idx, t_max_idx
                        else:
                            low_idx, high_idx = t_max_idx, t_min_idx
                        start_xyz = np.asarray(
                            [
                                float(s_x[low_idx]) / self.atar_pos_norm,
                                float(s_y[low_idx]) / self.atar_pos_norm,
                                float(s_z[low_idx]) / self.atar_pos_norm,
                            ],
                            dtype=np.float32,
                        )
                        stop_xyz = np.asarray(
                            [
                                float(s_x[high_idx]) / self.atar_pos_norm,
                                float(s_y[high_idx]) / self.atar_pos_norm,
                                float(s_z[high_idx]) / self.atar_pos_norm,
                            ],
                            dtype=np.float32,
                        )
                    else:
                        start_xyz = np.zeros((3,), dtype=np.float32)
                        stop_xyz = np.zeros((3,), dtype=np.float32)
                    atar_slice_start_target_rows.append(start_xyz.reshape(1, 3))
                    atar_slice_stop_target_rows.append(stop_xyz.reshape(1, 3))

                theta_val = float(truth_theta[row_idx]) if truth_theta is not None and int(truth_theta.shape[0]) > int(row_idx) else 0.0
                phi_val = float(truth_phi[row_idx]) if truth_phi is not None and int(truth_phi.shape[0]) > int(row_idx) else 0.0
                angle_xyz = np.asarray(
                    [
                        np.sin(theta_val) * np.cos(phi_val),
                        np.sin(theta_val) * np.sin(phi_val),
                        np.cos(theta_val),
                    ],
                    dtype=np.float32,
                ).reshape(1, 3)
                pion_stop_xyz = np.asarray(
                    [
                        float(truth_pion_stop_x[row_idx]) if truth_pion_stop_x is not None and int(truth_pion_stop_x.shape[0]) > int(row_idx) else 0.0,
                        float(truth_pion_stop_y[row_idx]) if truth_pion_stop_y is not None and int(truth_pion_stop_y.shape[0]) > int(row_idx) else 0.0,
                        float(truth_pion_stop_z[row_idx]) if truth_pion_stop_z is not None and int(truth_pion_stop_z.shape[0]) > int(row_idx) else 0.0,
                    ],
                    dtype=np.float32,
                ).reshape(1, 3)
                pion_stop_xyz /= self.atar_pos_norm
                pion_stop_valid = bool(
                    truth_pion_stop_x is not None
                    and truth_pion_stop_y is not None
                    and truth_pion_stop_z is not None
                    and int(truth_pion_stop_x.shape[0]) > int(row_idx)
                    and int(truth_pion_stop_y.shape[0]) > int(row_idx)
                    and int(truth_pion_stop_z.shape[0]) > int(row_idx)
                    and np.isfinite(float(truth_pion_stop_x[row_idx]))
                    and np.isfinite(float(truth_pion_stop_y[row_idx]))
                    and np.isfinite(float(truth_pion_stop_z[row_idx]))
                )
                for _ in range(n_atar_slices):
                    atar_angle_target_rows.append(angle_xyz.copy())
                    atar_pion_stop_target_rows.append(pion_stop_xyz.copy())
                    atar_pion_stop_valid_target_rows.append(
                        np.asarray([1.0 if pion_stop_valid else 0.0], dtype=np.float32)
                    )
            graph_atar_slice_counts.append(n_atar_slices)

            if include_targets:
                if target_signal is not None:
                    target_value = float(target_signal[row_idx])
                elif target_energy is not None:
                    target_value = 1.0 if float(target_energy[row_idx]) > 0.0 else 0.0
                else:
                    target_value = 0.0
                graph_targets.append(target_value)

            node_base += n_nodes

        total_graphs = int(len(node_counts))

        if node_blocks:
            x_out = np.concatenate(node_blocks, axis=0).astype(np.float32, copy=False)
        else:
            x_out = np.zeros((0, self.node_feature_dim), dtype=np.float32)

        if edge_attr_blocks:
            edge_attr_out = np.concatenate(edge_attr_blocks, axis=0).astype(np.float32, copy=False)
        else:
            edge_attr_out = np.zeros((0, self.edge_feature_dim), dtype=np.float32)

        if edge_index_blocks:
            edge_index_out = np.concatenate(edge_index_blocks, axis=1).astype(np.int64, copy=False)
        else:
            edge_index_out = np.zeros((2, 0), dtype=np.int64)

        if node_slice_id_blocks:
            node_slice_id_out = np.concatenate(node_slice_id_blocks, axis=0).astype(np.int64, copy=False)
        else:
            node_slice_id_out = np.zeros((0,), dtype=np.int64)

        total_slices = int(len(slice_graph_ids))
        slice_graph_id_out = np.asarray(slice_graph_ids, dtype=np.int64)
        slice_ptr_out = np.zeros((total_slices + 1,), dtype=np.int64)
        if total_slices > 0:
            slice_ptr_out[1:] = np.cumsum(np.asarray(slice_node_counts, dtype=np.int64), dtype=np.int64)

        graph_slice_ptr_out = np.zeros((total_graphs + 1,), dtype=np.int64)
        if total_graphs > 0:
            graph_slice_ptr_out[1:] = np.cumsum(np.asarray(graph_slice_counts, dtype=np.int64), dtype=np.int64)

        atar_slice_ptr_out = np.zeros((total_graphs + 1,), dtype=np.int64)
        if total_graphs > 0:
            atar_slice_ptr_out[1:] = np.cumsum(np.asarray(graph_atar_slice_counts, dtype=np.int64), dtype=np.int64)

        node_ptr = np.zeros((total_graphs + 1,), dtype=np.int64)
        edge_ptr = np.zeros((total_graphs + 1,), dtype=np.int64)
        if total_graphs > 0:
            node_ptr[1:] = np.cumsum(np.asarray(node_counts, dtype=np.int64), dtype=np.int64)
            edge_ptr[1:] = np.cumsum(np.asarray(edge_counts, dtype=np.int64), dtype=np.int64)

        state["layout"] = {
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "total_graphs": int(total_graphs),
        }
        state["x_out"] = x_out
        state["edge_attr_out"] = edge_attr_out
        state["edge_index_out"] = edge_index_out
        state["graph_event_id"] = np.asarray(graph_event_ids, dtype=np.int64)
        state["graph_time_group_id"] = np.zeros((total_graphs,), dtype=np.int64)
        state["node_slice_id_out"] = node_slice_id_out
        state["slice_graph_id_out"] = slice_graph_id_out
        state["slice_ptr_out"] = slice_ptr_out
        state["graph_slice_ptr_out"] = graph_slice_ptr_out
        state["atar_slice_ptr_out"] = atar_slice_ptr_out

        if atar_node_pdg_target_blocks:
            state["atar_node_pdg_target_out"] = np.concatenate(atar_node_pdg_target_blocks, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["atar_node_pdg_target_out"] = np.zeros((0, 3), dtype=np.float32)

        if atar_true_event_id_blocks:
            state["atar_true_event_id_out"] = np.concatenate(atar_true_event_id_blocks, axis=0).astype(
                np.int64, copy=False
            )
        else:
            state["atar_true_event_id_out"] = np.zeros((0,), dtype=np.int64)

        if is_trigger_target_blocks:
            state["is_trigger_target_out"] = np.concatenate(is_trigger_target_blocks, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["is_trigger_target_out"] = np.zeros((0,), dtype=np.float32)

        if lyso_fracs_target_blocks:
            state["lyso_fracs_target_out"] = np.concatenate(lyso_fracs_target_blocks, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["lyso_fracs_target_out"] = np.zeros((0, self.MAX_LYSO_OBJECTS), dtype=np.float32)

        if lyso_payload_target_rows:
            state["lyso_payload_target_out"] = np.stack(lyso_payload_target_rows, axis=0).astype(np.float32, copy=False)
        else:
            state["lyso_payload_target_out"] = np.zeros((0, self.MAX_LYSO_OBJECTS, 4), dtype=np.float32)

        if lyso_mask_target_rows:
            state["lyso_mask_target_out"] = np.stack(lyso_mask_target_rows, axis=0).astype(np.float32, copy=False)
        else:
            state["lyso_mask_target_out"] = np.zeros((0, self.MAX_LYSO_OBJECTS), dtype=np.float32)

        if atar_slice_pdg_target_rows:
            state["atar_slice_pdg_target_out"] = np.concatenate(atar_slice_pdg_target_rows, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["atar_slice_pdg_target_out"] = np.zeros((0, 3), dtype=np.float32)

        if atar_slice_multi_target_rows:
            state["atar_slice_multi_target_out"] = np.concatenate(atar_slice_multi_target_rows, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["atar_slice_multi_target_out"] = np.zeros((0,), dtype=np.float32)

        if atar_slice_trigger_target_rows:
            state["atar_slice_trigger_target_out"] = np.concatenate(atar_slice_trigger_target_rows, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["atar_slice_trigger_target_out"] = np.zeros((0,), dtype=np.float32)

        if atar_slice_start_target_rows:
            state["atar_slice_start_target_out"] = np.concatenate(atar_slice_start_target_rows, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["atar_slice_start_target_out"] = np.zeros((0, 3), dtype=np.float32)

        if atar_slice_stop_target_rows:
            state["atar_slice_stop_target_out"] = np.concatenate(atar_slice_stop_target_rows, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["atar_slice_stop_target_out"] = np.zeros((0, 3), dtype=np.float32)

        if atar_angle_target_rows:
            state["atar_angle_target_out"] = np.concatenate(atar_angle_target_rows, axis=0).astype(np.float32, copy=False)
        else:
            state["atar_angle_target_out"] = np.zeros((0, 3), dtype=np.float32)

        if atar_pion_stop_target_rows:
            state["atar_pion_stop_target_out"] = np.concatenate(atar_pion_stop_target_rows, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["atar_pion_stop_target_out"] = np.zeros((0, 3), dtype=np.float32)
        if atar_pion_stop_valid_target_rows:
            state["atar_pion_stop_valid_target_out"] = np.concatenate(
                atar_pion_stop_valid_target_rows,
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            state["atar_pion_stop_valid_target_out"] = np.zeros((0,), dtype=np.float32)

        if positron_energy_target_rows:
            state["positron_initial_energy_target_out"] = np.stack(positron_energy_target_rows, axis=0).astype(
                np.float32, copy=False
            )
        else:
            state["positron_initial_energy_target_out"] = np.zeros((0, 1), dtype=np.float32)

        state["has_trigger_positron_out"] = np.asarray(has_trigger_positron_rows, dtype=np.float32)

        if include_targets:
            y_graph = np.asarray(graph_targets, dtype=np.float32).reshape(-1, 1)
            state["y_graph_out"] = y_graph
            state["y_event_out"] = y_graph
            state["y_slice_out"] = np.asarray(slice_targets, dtype=np.float32).reshape(-1, 1)
        else:
            state["y_graph_out"] = np.zeros((0, 1), dtype=np.float32)
            state["y_event_out"] = np.zeros((0, 1), dtype=np.float32)
            state["y_slice_out"] = np.zeros((0, 1), dtype=np.float32)

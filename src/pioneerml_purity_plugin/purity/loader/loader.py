from __future__ import annotations

from typing import Any

import numpy as np

from pioneerml.data_loader.loaders.array_store import NDArrayColumnSpec
from pioneerml.data_loader.loaders.array_store.schemas import FeatureSchema, LoaderSchema, TargetSchema
from pioneerml.data_loader.loaders.config import DataFlowConfig, GraphTensorDims, SplitSampleConfig
from pioneerml.data_loader.loaders.factory.registry import REGISTRY as LOADER_REGISTRY
from pioneerml.data_loader.loaders.input_source import InputBackend, InputSourceSet, create_input_backend
from pioneerml.data_loader.loaders.stage.stages import (
    BaseStage,
    BatchPackStage,
    DistributedShardStage,
    ExtractFeaturesStage,
    RowFilterStage,
    RowJoinStage,
)
from pioneerml.staged_runtime.stage_observers import StageObserver

from .multi_level_graph_loader import MultiLevelGraphLoader
from .stages import BuildPurityGraphStage


@LOADER_REGISTRY.register("purity")
class PurityGraphLoader(MultiLevelGraphLoader):
    """Structured staged loader for Omar-style PURITY event parquet."""

    NODE_FEATURE_DIM = 10
    EDGE_FEATURE_DIM = 11
    TARGET_DIM = 1

    @classmethod
    def from_factory(
        cls,
        *,
        input_sources: InputSourceSet,
        input_backend_name: str,
        mode: str,
        data_flow_config: DataFlowConfig,
        split_config: SplitSampleConfig,
        loader_params: dict[str, Any] | None = None,
    ):
        params = dict(loader_params or {})
        stage_overrides = params.get("stage_overrides")
        stage_observer = params.get("stage_observer")
        profiling = dict(params.get("profiling") or {})
        loader = cls(
            input_sources=input_sources,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            input_backend=params.get("input_backend"),
            input_backend_name=input_backend_name,
            atar_pos_norm=float(params.get("atar_pos_norm", 10.0)),
            atar_energy_norm=float(params.get("atar_energy_norm", 1.0)),
            atar_time_norm=float(params.get("atar_time_norm", 500.0)),
            lyso_pos_norm=float(params.get("lyso_pos_norm", 100.0)),
            lyso_energy_norm=float(params.get("lyso_energy_norm", 70.0)),
            lyso_time_norm=float(params.get("lyso_time_norm", 500.0)),
            stage_overrides=stage_overrides if isinstance(stage_overrides, dict) else None,
            stage_observer=stage_observer if isinstance(stage_observer, StageObserver) else None,
            profiling=profiling,
        )
        return cls._apply_common_loader_params(loader=loader, loader_params=params)

    def __init__(
        self,
        input_sources: InputSourceSet,
        *,
        mode: str = MultiLevelGraphLoader.MODE_TRAIN,
        input_backend: InputBackend | None = None,
        input_backend_name: str = "parquet",
        data_flow_config: DataFlowConfig | None = None,
        split_config: SplitSampleConfig | None = None,
        graph_dims: GraphTensorDims | None = None,
        atar_pos_norm: float = 10.0,
        atar_energy_norm: float = 1.0,
        atar_time_norm: float = 500.0,
        lyso_pos_norm: float = 100.0,
        lyso_energy_norm: float = 70.0,
        lyso_time_norm: float = 500.0,
        stage_overrides: dict[str, BaseStage] | None = None,
        stage_observer: StageObserver | None = None,
        profiling: dict | None = None,
    ) -> None:
        self._resolved_field_specs: tuple[NDArrayColumnSpec, ...] = ()
        self.atar_pos_norm = float(atar_pos_norm)
        self.atar_energy_norm = float(atar_energy_norm)
        self.atar_time_norm = float(atar_time_norm)
        self.lyso_pos_norm = float(lyso_pos_norm)
        self.lyso_energy_norm = float(lyso_energy_norm)
        self.lyso_time_norm = float(lyso_time_norm)

        self.graph_dims = graph_dims or GraphTensorDims(
            node_feature_dim=int(self.NODE_FEATURE_DIM),
            edge_feature_dim=int(self.EDGE_FEATURE_DIM),
            graph_target_dim=int(self.TARGET_DIM),
        )
        self.schema = self.input_schema()

        include_targets = str(mode).strip().lower() != str(self.MODE_INFERENCE).lower()
        resolved_input_sources = input_sources
        resolved_input_backend = input_backend if input_backend is not None else create_input_backend(input_backend_name)
        declared_specs = self.schema.to_column_specs(include_targets=True)
        self._resolved_field_specs = resolved_input_backend.resolve_declared_field_specs(
            input_sources=resolved_input_sources,
            field_specs=declared_specs,
            include_targets=include_targets,
        )

        super().__init__(
            input_sources=resolved_input_sources,
            input_backend=resolved_input_backend,
            resolved_field_specs=self._resolved_field_specs,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            stage_overrides=stage_overrides,
            stage_observer=stage_observer,
            profiling=profiling,
        )

        required = self.required_fields(include_targets=self.include_targets)
        missing = [c for c in required if c not in self.main_fields]
        if missing:
            raise ValueError(f"Missing required columns for mode={self.mode}: {missing}")

    def input_schema(self) -> LoaderSchema:
        features = FeatureSchema(
            fields=(
                NDArrayColumnSpec(column="event_id", field="event_id", dtype=np.int64, target_only=False),
                NDArrayColumnSpec(column="atar_x", field="atar_x", dtype=np.float32, target_only=False),
                NDArrayColumnSpec(column="atar_y", field="atar_y", dtype=np.float32, target_only=False),
                NDArrayColumnSpec(column="atar_z", field="atar_z", dtype=np.float32, target_only=False),
                NDArrayColumnSpec(column="atar_E", field="atar_E", dtype=np.float32, target_only=False),
                NDArrayColumnSpec(column="atar_t", field="atar_t", dtype=np.float32, target_only=False),
                NDArrayColumnSpec(column="atar_view", field="atar_view", dtype=np.int32, target_only=False),
                NDArrayColumnSpec(column="atar_slice", field="atar_slice", dtype=np.int32, required=False, target_only=False),
                NDArrayColumnSpec(
                    column="atar_slice_mean_t",
                    field="atar_slice_mean_t",
                    dtype=np.float32,
                    required=False,
                    target_only=False,
                ),
                NDArrayColumnSpec(
                    column="atar_slice_id",
                    field="atar_slice_id",
                    dtype=np.int32,
                    required=False,
                    target_only=False,
                ),
                NDArrayColumnSpec(column="lyso_x", field="lyso_x", dtype=np.float32, required=False, target_only=False),
                NDArrayColumnSpec(column="lyso_y", field="lyso_y", dtype=np.float32, required=False, target_only=False),
                NDArrayColumnSpec(column="lyso_z", field="lyso_z", dtype=np.float32, required=False, target_only=False),
                NDArrayColumnSpec(column="lyso_E", field="lyso_E", dtype=np.float32, required=False, target_only=False),
                NDArrayColumnSpec(column="lyso_t", field="lyso_t", dtype=np.float32, required=False, target_only=False),
                NDArrayColumnSpec(column="lyso_slice", field="lyso_slice", dtype=np.int32, required=False, target_only=False),
                NDArrayColumnSpec(
                    column="lyso_slice_mean_t",
                    field="lyso_slice_mean_t",
                    dtype=np.float32,
                    required=False,
                    target_only=False,
                ),
            )
        )
        targets = TargetSchema(
            fields=(
                NDArrayColumnSpec(
                    column="truth_is_signal",
                    field="truth_is_signal",
                    dtype=np.float32,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="truth_positron_energy",
                    field="truth_positron_energy",
                    dtype=np.float32,
                    required=True,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="truth_theta",
                    field="truth_theta",
                    dtype=np.float32,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="truth_phi",
                    field="truth_phi",
                    dtype=np.float32,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="truth_pion_stop_x",
                    field="truth_pion_stop_x",
                    dtype=np.float32,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="truth_pion_stop_y",
                    field="truth_pion_stop_y",
                    dtype=np.float32,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="truth_pion_stop_z",
                    field="truth_pion_stop_z",
                    dtype=np.float32,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="atar_pdg",
                    field="atar_pdg",
                    dtype=np.int64,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="atar_origin",
                    field="atar_origin",
                    dtype=np.int64,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="atar_truth_t",
                    field="atar_truth_t",
                    dtype=np.float32,
                    required=False,
                    target_only=True,
                ),
                NDArrayColumnSpec(
                    column="lyso_origin",
                    field="lyso_origin",
                    dtype=np.int64,
                    required=False,
                    target_only=True,
                ),
            )
        )
        return LoaderSchema(features=features, targets=targets)

    def default_stage_order(self) -> list[str]:
        return [
            "row_filter",
            "distributed_shard",
            "row_join",
            "extract_features",
            "build_purity_graph",
            "pack_batch",
        ]

    def default_stages(self) -> dict[str, BaseStage]:
        return {
            "row_filter": RowFilterStage(
                event_id_column="event_id",
                split_config=self.split_config,
            ),
            "distributed_shard": DistributedShardStage(event_id_column="event_id"),
            "row_join": RowJoinStage(),
            "extract_features": ExtractFeaturesStage(
                column_specs=self.schema.to_column_specs(include_targets=True),
                output_state_key="features_in",
            ),
            "build_purity_graph": BuildPurityGraphStage(
                input_state_key="features_in",
                node_feature_dim=int(self.empty_node_feature_dim()),
                edge_feature_dim=int(self.empty_edge_feature_dim()),
                atar_pos_norm=float(self.atar_pos_norm),
                atar_energy_norm=float(self.atar_energy_norm),
                atar_time_norm=float(self.atar_time_norm),
                lyso_pos_norm=float(self.lyso_pos_norm),
                lyso_energy_norm=float(self.lyso_energy_norm),
                lyso_time_norm=float(self.lyso_time_norm),
                cache_templates=None,
                cache_max_entries=None,
            ),
            "pack_batch": BatchPackStage(
                tensor_state_fields={
                    "x_node": "x_out",
                    "x_edge": "edge_attr_out",
                    "edge_index": "edge_index_out",
                    "graph_event_id": "graph_event_id",
                    "graph_time_group_id": "graph_time_group_id",
                    "node_slice_id": "node_slice_id_out",
                    "slice_graph_id": "slice_graph_id_out",
                    "slice_ptr": "slice_ptr_out",
                    "graph_slice_ptr": "graph_slice_ptr_out",
                },
                tensor_layout_fields={
                    "node_ptr": "node_ptr",
                    "edge_ptr": "edge_ptr",
                },
                scalar_state_fields={"num_rows": "n_rows"},
                scalar_layout_fields={"num_graphs": "total_graphs"},
                optional_tensor_state_fields={
                    "y_graph": "y_graph_out",
                    "y_event": "y_event_out",
                    "y_slice": "y_slice_out",
                    "atar_slice_ptr": "atar_slice_ptr_out",
                    "atar_node_pdg_target": "atar_node_pdg_target_out",
                    "atar_true_event_id": "atar_true_event_id_out",
                    "atar_slice_pdg_target": "atar_slice_pdg_target_out",
                    "atar_slice_multi_target": "atar_slice_multi_target_out",
                    "atar_slice_trigger_target": "atar_slice_trigger_target_out",
                    "atar_slice_start_target": "atar_slice_start_target_out",
                    "atar_slice_stop_target": "atar_slice_stop_target_out",
                    "atar_angle_target": "atar_angle_target_out",
                    "atar_pion_stop_target": "atar_pion_stop_target_out",
                    "positron_initial_energy_target": "positron_initial_energy_target_out",
                    "lyso_fracs_target": "lyso_fracs_target_out",
                    "lyso_payload_target": "lyso_payload_target_out",
                    "lyso_mask_target": "lyso_mask_target_out",
                    "is_trigger_target": "is_trigger_target_out",
                    "has_trigger_positron": "has_trigger_positron_out",
                },
            ),
        }

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from pioneerml.data_writer.array_store import OutputColumnSpec, OutputSchema
from pioneerml.data_writer.factory.registry import REGISTRY as WRITER_REGISTRY
from pioneerml.data_writer.input_source import TimeGroupPredictionSet
from pioneerml.data_writer.stage.stages import (
    AppendChunkStage,
    BufferChunkStage,
    CloseSinksStage,
    EmitRunOutputsStage,
    InitRunStateStage,
    OpenSinksStage,
    ResolveIndexingStage,
    ValidateInputsStage,
)
from pioneerml.data_writer.structured.graph.time_group.stages import (
    FinalizeBufferedWritesStage,
    StitchTimeGroupAlignedStructureStage,
)
from pioneerml.data_writer.structured.graph.time_group.time_group_graph_data_writer import TimeGroupGraphDataWriter
from pioneerml.data_writer.structured.structured_data_writer import WriterPhaseOrder, WriterPhaseStages


@WRITER_REGISTRY.register("purity")
class PurityDataWriter(TimeGroupGraphDataWriter):
    """Writer for PURITY event-level signal predictions."""

    FAILURE_NONE = 0
    FAILURE_NO_MODEL_OUTPUT = 1
    FAILURE_INPUT_CONVERSION = 2
    FAILURE_MODEL_EXECUTION = 3
    FAILURE_PREDICTION_CONVERSION = 4
    FAILURE_LOADER = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inference_failures_by_source: dict[str, dict[int, tuple[int, str]]] = {}
        self._loader_failures_by_source: dict[str, tuple[int, str]] = {}
        self._inference_success_by_source: dict[str, set[int]] = {}
        self._output_source_context: str | None = None

    def on_start(self, *, state):
        self._inference_failures_by_source.clear()
        self._loader_failures_by_source.clear()
        self._inference_success_by_source.clear()
        self._output_source_context = None
        return super().on_start(state=state)

    def set_output_source_context(self, src_path) -> None:
        self._output_source_context = (
            None if src_path is None else str(Path(src_path).expanduser().resolve())
        )

    @staticmethod
    def _failure_text(*, stage: str, error: BaseException, limit: int = 512) -> str:
        message = " ".join(str(error).split())
        text = f"{stage}:{error.__class__.__name__}"
        if message:
            text += f":{message}"
        return text[: int(limit)]

    @classmethod
    def _failure_code(cls, stage: str) -> int:
        return {
            "input_conversion": cls.FAILURE_INPUT_CONVERSION,
            "model_execution": cls.FAILURE_MODEL_EXECUTION,
            "prediction_conversion": cls.FAILURE_PREDICTION_CONVERSION,
            "loader": cls.FAILURE_LOADER,
        }.get(str(stage), cls.FAILURE_MODEL_EXECUTION)

    def record_inference_success(self, *, src_path, event_ids) -> None:
        src_key = str(Path(src_path).expanduser().resolve())
        successes = self._inference_success_by_source.setdefault(src_key, set())
        successes.update(
            int(event_id)
            for event_id in np.asarray(event_ids, dtype=np.int64).reshape(-1).tolist()
        )

    def record_inference_failure(
        self,
        *,
        src_path,
        event_ids,
        stage: str,
        error: BaseException,
    ) -> None:
        src_key = str(Path(src_path).expanduser().resolve())
        failures = self._inference_failures_by_source.setdefault(src_key, {})
        failure = (self._failure_code(stage), self._failure_text(stage=stage, error=error))
        for event_id in np.asarray(event_ids, dtype=np.int64).reshape(-1).tolist():
            failures[int(event_id)] = failure

    def record_loader_failure(self, *, src_path, error: BaseException) -> None:
        src_key = str(Path(src_path).expanduser().resolve())
        self._loader_failures_by_source[src_key] = (
            self.FAILURE_LOADER,
            self._failure_text(stage="loader", error=error),
        )

    def output_schema(self) -> OutputSchema:
        def output_column(output_column: str, **kwargs) -> OutputColumnSpec:
            metadata = dict(kwargs.pop("metadata", {}))
            metadata["pioneer.tes_path"] = f"/Event/ML/Purity/{output_column}"
            return OutputColumnSpec(output_column, metadata=metadata, **kwargs)

        return OutputSchema(
            fields=(
                output_column(
                    "event_id",
                    value_type=pa.int64(),
                    metadata={"pioneer.role": "alignment"},
                ),
                output_column(
                    "time_group_ids",
                    value_type=pa.int64(),
                ),
                output_column(
                    "pred_purity_signal",
                    model_output_name="main",
                    output_index=0,
                    dtype=np.float32,
                    value_type=pa.float32(),
                ),
                output_column(
                    "pred_purity_logit",
                    model_output_name="logit",
                    output_index=0,
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                output_column(
                    "pred_purity_summary_accepted",
                    model_output_name="summary_accepted",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                output_column(
                    "pred_purity_summary_positron_energy",
                    model_output_name="summary_positron_energy",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                output_column(
                    "pred_purity_summary_positron_time",
                    model_output_name="summary_positron_time",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                output_column(
                    "pred_purity_summary_positron_polar_angle",
                    model_output_name="summary_positron_polar_angle",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                output_column(
                    "pred_purity_guard_valid",
                    model_output_name="guard_valid",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                output_column(
                    "pred_purity_guard_reason",
                    model_output_name="guard_reason",
                    dtype=np.int32,
                    value_type=pa.int32(),
                    required=False,
                    metadata={
                        "pioneer.guard_reason_codes": (
                            "0=valid,1=no_hits,2=no_atar_hits,3=non_finite_logit,"
                            "4=token_graph_index_oob,5=internal_model_numerical_failure,"
                            "6=general_inference_failure"
                        )
                    },
                ),
                output_column("inference_success", value_type=pa.bool_()),
                output_column(
                    "inference_failure_code",
                    value_type=pa.int32(),
                    metadata={
                        "pioneer.inference_failure_codes": (
                            "0=success,1=no_model_output,2=input_conversion,3=model_execution,"
                            "4=prediction_conversion,5=loader"
                        )
                    },
                ),
                output_column("inference_failure_reason", value_type=pa.string()),
            )
        )

    def build_inference_failure_prediction_set(
        self,
        *,
        batch,
        src_path,
        num_rows: int,
        stage: str,
        error: BaseException,
        cfg: dict | None = None,
    ) -> TimeGroupPredictionSet:
        _ = cfg
        event_ids = getattr(batch, "graph_event_id", None)
        if not isinstance(event_ids, torch.Tensor):
            raise RuntimeError("PURITY inference fallback requires batch.graph_event_id.") from error
        event_ids_np = event_ids.detach().cpu().to(torch.int64).numpy().reshape(-1)
        group_ids = getattr(batch, "graph_time_group_id", None)
        if isinstance(group_ids, torch.Tensor) and int(group_ids.numel()) == int(event_ids.numel()):
            group_ids_np = group_ids.detach().cpu().to(torch.int64).numpy().reshape(-1)
        else:
            group_ids_np = np.zeros(event_ids_np.shape, dtype=np.int64)
        count = int(event_ids_np.shape[0])
        guard_reason = 5 if stage == "model_execution" and "non-finite" in str(error).lower() else 6
        return TimeGroupPredictionSet(
            src_path=src_path,
            prediction_event_ids_np=event_ids_np,
            model_outputs_by_name={
                "main": np.full((count, 1), 0.5, dtype=np.float32),
                "logit": np.zeros((count, 1), dtype=np.float32),
                "summary_accepted": np.zeros(count, dtype=np.float32),
                "summary_positron_energy": np.zeros(count, dtype=np.float32),
                "summary_positron_time": np.zeros(count, dtype=np.float32),
                "summary_positron_polar_angle": np.zeros(count, dtype=np.float32),
                "guard_valid": np.zeros(count, dtype=np.float32),
                "guard_reason": np.full(count, guard_reason, dtype=np.int32),
            },
            num_rows=int(num_rows),
            time_group_event_ids_np=event_ids_np,
            time_group_ids_np=group_ids_np,
        )

    def prepare_output_table(self, table: pa.Table) -> pa.Table:
        if "event_id" not in table.column_names:
            return super().prepare_output_table(table)
        event_ids = np.asarray(
            table.column("event_id").combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        src_key = self._output_source_context or ""
        failures = self._inference_failures_by_source.get(src_key, {})
        successes = self._inference_success_by_source.get(src_key, set())
        source_default = self._loader_failures_by_source.get(src_key)

        success: list[bool] = []
        codes: list[int] = []
        reasons: list[str] = []
        for event_id in event_ids.tolist():
            explicit = failures.get(int(event_id))
            if explicit is not None:
                code, reason = explicit
            elif int(event_id) in successes:
                code, reason = self.FAILURE_NONE, ""
            elif source_default is not None:
                code, reason = source_default
            else:
                code, reason = self.FAILURE_NO_MODEL_OUTPUT, "no_model_output"
            success.append(int(code) == self.FAILURE_NONE)
            codes.append(int(code))
            reasons.append(str(reason))

        table = table.append_column("inference_success", pa.array(success, type=pa.bool_()))
        table = table.append_column("inference_failure_code", pa.array(codes, type=pa.int32()))
        table = table.append_column("inference_failure_reason", pa.array(reasons, type=pa.string()))
        return super().prepare_output_table(table)

    def default_stage_order(self) -> WriterPhaseOrder:
        return WriterPhaseOrder(
            start=["init_run_state", "open_sinks"],
            chunk=[
                "validate_inputs",
                "resolve_indexing",
                "stitch_structure",
                "append_chunk",
                "buffer_chunk",
            ],
            finalize=["finalize_buffered_writes", "close_sinks", "emit_run_outputs"],
        )

    def default_stages(self) -> WriterPhaseStages:
        return WriterPhaseStages(
            start={
                "init_run_state": InitRunStateStage(),
                "open_sinks": OpenSinksStage(),
            },
            chunk={
                "validate_inputs": ValidateInputsStage(
                    required_state_keys=(
                        "src_path",
                        "prediction_event_ids_np",
                        "prediction_columns",
                        "num_rows",
                        "output_dir",
                        "time_group_event_ids_np",
                        "time_group_ids_np",
                    ),
                ),
                "resolve_indexing": ResolveIndexingStage(
                    index_keys=("prediction_event_ids_np", "time_group_event_ids_np", "time_group_ids_np")
                ),
                "stitch_structure": StitchTimeGroupAlignedStructureStage(
                    prediction_event_ids_key="prediction_event_ids_np",
                    time_group_event_ids_key="time_group_event_ids_np",
                    time_group_ids_key="time_group_ids_np",
                ),
                "append_chunk": AppendChunkStage(),
                "buffer_chunk": BufferChunkStage(),
            },
            finalize={
                "finalize_buffered_writes": FinalizeBufferedWritesStage(),
                "close_sinks": CloseSinksStage(),
                "emit_run_outputs": EmitRunOutputsStage(),
            },
        )

    def build_prediction_set(
        self,
        *,
        batch,
        model_output,
        src_path,
        num_rows: int,
        cfg: dict | None = None,
    ) -> TimeGroupPredictionSet:
        _ = cfg
        logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        token_batch = None
        token_valid = None
        summary_tensors: dict[str, torch.Tensor] = {}
        if isinstance(logits, Mapping):
            logits_map = logits
            token_batch_maybe = logits_map.get("unified_token_batch")
            if isinstance(token_batch_maybe, torch.Tensor):
                token_batch = token_batch_maybe.to(dtype=torch.long)
            token_valid_maybe = logits_map.get("unified_token_valid")
            if isinstance(token_valid_maybe, torch.Tensor):
                token_valid = token_valid_maybe.to(dtype=torch.bool)
            preferred = logits_map.get("unified_event_logits")
            if isinstance(preferred, torch.Tensor):
                logits = preferred
            else:
                main = logits_map.get("main")
                if isinstance(main, torch.Tensor):
                    logits = main
            event_summary = logits_map.get("event_summary")
            if isinstance(event_summary, Mapping):
                accepted = event_summary.get("accepted")
                if isinstance(accepted, torch.Tensor):
                    summary_tensors["summary_accepted"] = accepted
                pos_e = event_summary.get("positron_energy")
                if isinstance(pos_e, torch.Tensor):
                    summary_tensors["summary_positron_energy"] = pos_e
                pos_t = event_summary.get("positron_time")
                if isinstance(pos_t, torch.Tensor):
                    summary_tensors["summary_positron_time"] = pos_t
                pos_ang = event_summary.get("positron_polar_angle")
                if isinstance(pos_ang, torch.Tensor):
                    summary_tensors["summary_positron_polar_angle"] = pos_ang
            # TorchScript path exports flattened summary tensors directly.
            for k in (
                "summary_accepted",
                "summary_positron_energy",
                "summary_positron_time",
                "summary_positron_polar_angle",
            ):
                v = logits_map.get(k)
                if isinstance(v, torch.Tensor):
                    summary_tensors[k] = v
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"{self.__class__.__name__} expected tensor logits or mapping containing "
                "'unified_event_logits'."
            )
        logits_np = logits.detach().cpu().to(torch.float32).numpy().astype("float32", copy=False)
        probs_np = torch.sigmoid(logits).detach().cpu().to(torch.float32).numpy().astype("float32", copy=False)

        graph_event_ids_np = batch.graph_event_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
        if hasattr(batch, "graph_time_group_id") and batch.graph_time_group_id is not None:
            graph_time_group_ids_np = batch.graph_time_group_id.to(torch.int64).cpu().numpy().astype("int64", copy=False)
        else:
            graph_time_group_ids_np = np.zeros((graph_event_ids_np.shape[0],), dtype=np.int64)

        prediction_event_ids_np = graph_event_ids_np
        prediction_group_ids_np = graph_time_group_ids_np
        token_index_np = None
        token_index_oob_mask: np.ndarray | None = None

        node_graph_id = getattr(batch, "node_graph_id", None)
        x_node = getattr(batch, "x_node", None)
        guard_valid_graph = np.ones((graph_event_ids_np.shape[0],), dtype=np.float32)
        guard_reason_graph = np.zeros((graph_event_ids_np.shape[0],), dtype=np.int32)
        if isinstance(node_graph_id, torch.Tensor) and isinstance(x_node, torch.Tensor) and int(graph_event_ids_np.shape[0]) > 0:
            node_graph_idx = node_graph_id.detach().cpu().numpy().astype(np.int64, copy=False)
            graph_count = int(graph_event_ids_np.shape[0])
            node_counts = np.bincount(node_graph_idx, minlength=graph_count).astype(np.int64, copy=False)
            is_atar = ((x_node[:, 5] > 0.5) | (x_node[:, 6] > 0.5)).detach().cpu().numpy().astype(np.bool_, copy=False)
            atar_counts = np.bincount(node_graph_idx[is_atar], minlength=graph_count).astype(np.int64, copy=False)
            no_hits = node_counts <= 0
            no_atar = atar_counts <= 0
            guard_valid_graph = (~(no_hits | no_atar)).astype(np.float32, copy=False)
            guard_reason_graph = np.where(no_hits, 1, np.where(no_atar, 2, 0)).astype(np.int32, copy=False)
        if isinstance(token_batch, torch.Tensor):
            idx = token_batch.detach().cpu().numpy().astype(np.int64, copy=False)
            if isinstance(token_valid, torch.Tensor):
                keep = token_valid.detach().cpu().numpy().astype(np.bool_, copy=False).reshape(-1)
                if keep.shape[0] == idx.shape[0]:
                    # TorchScript can emit token_batch/token_valid on the dense token axis
                    # while logits are already compacted to valid tokens. Handle both.
                    if int(logits.shape[0]) == int(idx.shape[0]):
                        idx = idx[keep]
                        probs_np = probs_np[keep]
                        logits_np = logits_np[keep]
                    elif int(logits.shape[0]) == int(keep.sum()):
                        idx = idx[keep]

            if int(idx.shape[0]) != int(logits.shape[0]):
                logits_rows = int(logits.shape[0])
                if int(idx.shape[0]) > logits_rows:
                    idx = idx[:logits_rows]
                else:
                    pad = np.zeros((logits_rows - int(idx.shape[0]),), dtype=np.int64)
                    idx = np.concatenate([idx, pad], axis=0)

            if idx.size > 0:
                num_graphs = int(graph_event_ids_np.shape[0])
                token_index_oob_mask = (idx < 0) | (idx >= num_graphs)
                if bool(np.any(token_index_oob_mask)):
                    idx = np.clip(idx, 0, max(num_graphs - 1, 0))
                prediction_event_ids_np = graph_event_ids_np[idx]
                prediction_group_ids_np = graph_time_group_ids_np[idx]
                token_index_np = idx
            else:
                prediction_event_ids_np = np.zeros((0,), dtype=np.int64)
                prediction_group_ids_np = np.zeros((0,), dtype=np.int64)
                logits_np = np.zeros((0, 1), dtype=np.float32)
                probs_np = np.zeros((0, 1), dtype=np.float32)
                token_index_np = np.zeros((0,), dtype=np.int64)

        model_outputs_by_name: dict[str, np.ndarray] = {
            "main": probs_np,
            "logit": logits_np,
        }
        if token_index_np is not None:
            model_outputs_by_name["token_graph_index"] = token_index_np

        if summary_tensors:
            num_graphs = int(graph_event_ids_np.shape[0])
            for key, tensor in summary_tensors.items():
                summary = tensor.detach().to(dtype=torch.float32)
                if summary.dim() > 1:
                    summary = summary.view(summary.shape[0], -1)
                    if int(summary.shape[1]) != 1:
                        continue
                    summary = summary[:, 0]
                if int(summary.numel()) != num_graphs:
                    continue
                if token_index_np is not None:
                    if int(token_index_np.size) == 0:
                        aligned = np.zeros((0,), dtype=np.float32)
                    else:
                        aligned = summary.cpu().numpy().astype("float32", copy=False)[token_index_np]
                else:
                    aligned = summary.cpu().numpy().astype("float32", copy=False)
                if int(aligned.shape[0]) == int(prediction_event_ids_np.shape[0]):
                    model_outputs_by_name[key] = aligned

        if token_index_np is not None:
            if int(token_index_np.size) == 0:
                guard_valid_pred = np.zeros((0,), dtype=np.float32)
                guard_reason_pred = np.zeros((0,), dtype=np.int32)
            else:
                guard_valid_pred = guard_valid_graph[token_index_np]
                guard_reason_pred = guard_reason_graph[token_index_np]
                if token_index_oob_mask is not None and token_index_oob_mask.shape[0] == guard_valid_pred.shape[0]:
                    guard_valid_pred = guard_valid_pred.copy()
                    guard_reason_pred = guard_reason_pred.copy()
                    guard_valid_pred[token_index_oob_mask] = 0.0
                    guard_reason_pred[token_index_oob_mask] = 4
        else:
            guard_valid_pred = guard_valid_graph
            guard_reason_pred = guard_reason_graph

        finite_mask_np = np.isfinite(logits_np.reshape(-1))
        if guard_valid_pred.shape[0] == finite_mask_np.shape[0]:
            guard_valid_pred = guard_valid_pred.copy()
            guard_reason_pred = guard_reason_pred.copy()
            guard_valid_pred[~finite_mask_np] = 0.0
            guard_reason_pred[~finite_mask_np] = 3
        if not bool(np.all(finite_mask_np)):
            logits_np = np.nan_to_num(logits_np, nan=0.0, posinf=0.0, neginf=0.0)
            probs_np = np.nan_to_num(probs_np, nan=0.5, posinf=1.0, neginf=0.0)

        model_outputs_by_name["main"] = probs_np
        model_outputs_by_name["logit"] = logits_np
        model_outputs_by_name["guard_valid"] = guard_valid_pred.astype(np.float32, copy=False)
        model_outputs_by_name["guard_reason"] = guard_reason_pred.astype(np.int32, copy=False)

        return TimeGroupPredictionSet(
            src_path=src_path,
            prediction_event_ids_np=prediction_event_ids_np,
            model_outputs_by_name=model_outputs_by_name,
            num_rows=int(num_rows),
            time_group_event_ids_np=prediction_event_ids_np,
            time_group_ids_np=prediction_group_ids_np,
        )

from __future__ import annotations

from collections.abc import Mapping

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

    def output_schema(self) -> OutputSchema:
        return OutputSchema(
            fields=(
                OutputColumnSpec(
                    "pred_purity_signal",
                    model_output_name="main",
                    output_index=0,
                    dtype=np.float32,
                    value_type=pa.float32(),
                ),
                OutputColumnSpec(
                    "pred_purity_logit",
                    model_output_name="logit",
                    output_index=0,
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                OutputColumnSpec(
                    "pred_purity_summary_accepted",
                    model_output_name="summary_accepted",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                OutputColumnSpec(
                    "pred_purity_summary_positron_energy",
                    model_output_name="summary_positron_energy",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                OutputColumnSpec(
                    "pred_purity_summary_positron_time",
                    model_output_name="summary_positron_time",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
                OutputColumnSpec(
                    "pred_purity_summary_positron_polar_angle",
                    model_output_name="summary_positron_polar_angle",
                    dtype=np.float32,
                    value_type=pa.float32(),
                    required=False,
                ),
            )
        )

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
            preferred = logits_map.get("unified_event_logits")
            if isinstance(preferred, torch.Tensor):
                logits = preferred
                token_batch_maybe = logits_map.get("unified_token_batch")
                if isinstance(token_batch_maybe, torch.Tensor):
                    token_batch = token_batch_maybe.to(dtype=torch.long)
                token_valid_maybe = logits_map.get("unified_token_valid")
                if isinstance(token_valid_maybe, torch.Tensor):
                    token_valid = token_valid_maybe.to(dtype=torch.bool)
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
        if isinstance(token_batch, torch.Tensor) and int(token_batch.numel()) == int(logits.shape[0]):
            idx = token_batch.detach().cpu().numpy().astype(np.int64, copy=False)
            if isinstance(token_valid, torch.Tensor):
                keep = token_valid.detach().cpu().numpy().astype(np.bool_, copy=False).reshape(-1)
                if keep.shape[0] == idx.shape[0]:
                    idx = idx[keep]
                    probs_np = probs_np[keep]
                    logits_np = logits_np[keep]
            if idx.size > 0:
                if int(idx.min()) < 0 or int(idx.max()) >= int(graph_event_ids_np.shape[0]):
                    raise ValueError(
                        f"{self.__class__.__name__} got unified_token_batch index out of range: "
                        f"max={int(idx.max())}, num_graphs={int(graph_event_ids_np.shape[0])}."
                    )
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

        return TimeGroupPredictionSet(
            src_path=src_path,
            prediction_event_ids_np=prediction_event_ids_np,
            model_outputs_by_name=model_outputs_by_name,
            num_rows=int(num_rows),
            time_group_event_ids_np=prediction_event_ids_np,
            time_group_ids_np=prediction_group_ids_np,
        )

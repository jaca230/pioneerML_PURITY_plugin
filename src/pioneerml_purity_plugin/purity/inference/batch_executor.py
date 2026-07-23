from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch

from pioneerml.data_writer.input_source import PredictionSet
from pioneerml.inference import (
    BaseInferenceBatchExecutor,
    InferenceBatchContext,
    InferenceFailure,
)
from pioneerml.inference.batch_executor.factory.registry import REGISTRY


LOGGER = logging.getLogger(__name__)


@REGISTRY.register("purity")
class PurityInferenceBatchExecutor(BaseInferenceBatchExecutor):
    """PURITY inference recovery with batch marking or event isolation."""

    ERROR_POLICIES = frozenset({"raise", "mark_batch", "isolate_events"})

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        policy = str(self.config.get("error_policy", "raise")).strip().lower()
        if policy not in self.ERROR_POLICIES:
            allowed = ", ".join(sorted(self.ERROR_POLICIES))
            raise ValueError(f"Invalid PURITY inference error_policy={policy!r}; expected one of: {allowed}.")
        self.error_policy = policy

    @staticmethod
    def _event_ids(*, context: InferenceBatchContext) -> list[int]:
        resolver = getattr(context.loader, "inference_event_ids", None)
        if not callable(resolver):
            return []
        return [int(value) for value in resolver(context.batch)]

    @staticmethod
    def _require_recovery_writer(context: InferenceBatchContext) -> None:
        required = (
            "record_inference_success",
            "record_inference_failure",
            "record_loader_failure",
            "build_inference_failure_prediction_set",
        )
        missing = [name for name in required if not hasattr(context.writer, name)]
        if missing:
            raise RuntimeError(
                f"{context.writer.__class__.__name__} does not support PURITY inference recovery; "
                f"missing methods: {missing}."
            )

    def handle_success(
        self,
        *,
        context: InferenceBatchContext,
        prediction_set: PredictionSet,
    ) -> None:
        _ = prediction_set
        self._require_recovery_writer(context)
        event_ids = self._event_ids(context=context)
        if not event_ids:
            raise RuntimeError("PURITY inference requires event IDs for output status tracking.")
        context.writer.record_inference_success(
            src_path=context.source_path,
            event_ids=event_ids,
        )

    def handle_failure(
        self,
        *,
        context: InferenceBatchContext,
        failure: InferenceFailure,
    ) -> PredictionSet | None:
        if self.error_policy == "raise":
            raise failure.error
        self._require_recovery_writer(context)
        if isinstance(failure.error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()

        event_ids = self._event_ids(context=context)
        if not event_ids:
            raise RuntimeError("Cannot recover PURITY inference failure without event IDs.") from failure.error

        if self.error_policy == "isolate_events":
            splitter = getattr(context.loader, "split_inference_batch", None)
            split = splitter(context.batch) if callable(splitter) else None
            if split is not None:
                LOGGER.warning(
                    "PURITY inference %s failed; splitting batch of %d event(s), first event_ids=%s: %s",
                    failure.stage.value,
                    len(event_ids),
                    event_ids[:20],
                    failure.error,
                )
                for sub_batch in split:
                    self.execute_batch(context=context.with_batch(sub_batch))
                return None

        context.writer.record_inference_failure(
            src_path=context.source_path,
            event_ids=event_ids,
            stage=failure.stage.value,
            error=failure.error,
        )
        prediction_set = context.writer.build_inference_failure_prediction_set(
            batch=context.batch,
            src_path=context.source_path,
            num_rows=context.source_num_rows,
            stage=failure.stage.value,
            error=failure.error,
            cfg=context.config,
        )
        if not isinstance(prediction_set, PredictionSet):
            raise RuntimeError(
                f"{context.writer.__class__.__name__}.build_inference_failure_prediction_set(...) "
                "must return PredictionSet."
            ) from failure.error
        scope = "event" if len(event_ids) == 1 else "batch"
        LOGGER.error(
            "PURITY inference %s failed; marking %s of %d event(s) invalid, first event_ids=%s: %s",
            failure.stage.value,
            scope,
            len(event_ids),
            event_ids[:20],
            failure.error,
        )
        return prediction_set

    def handle_loader_failure(
        self,
        *,
        context: InferenceBatchContext,
        error: Exception,
    ) -> None:
        if self.error_policy == "raise":
            raise error
        self._require_recovery_writer(context)
        context.writer.record_loader_failure(src_path=context.source_path, error=error)
        LOGGER.error(
            "PURITY inference loader failed for source=%s; remaining events will be marked invalid: %s",
            context.source_path,
            error,
        )

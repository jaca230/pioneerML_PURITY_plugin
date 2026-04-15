from typing import Any

from zenml import pipeline, step

from .steps import (
    PurityEvaluationStep,
    PurityExportStep,
    PurityHPOStep,
    PurityStagedTrainingStep,
)


@step(name="tune_model", enable_cache=False)
def tune_model_step(pipeline_config: dict | None = None) -> Any:
    return PurityHPOStep(pipeline_config=pipeline_config).execute()


@step(name="train_model", enable_cache=False)
def train_model_step(
    hpo_payload: Any = None,
    pipeline_config: dict | None = None,
) -> Any:
    return PurityStagedTrainingStep(pipeline_config=pipeline_config).execute(
        payloads={"hpo": hpo_payload},
    )


@step(name="evaluate_model", enable_cache=False)
def evaluate_model_step(
    train_payload: Any,
    pipeline_config: dict | None = None,
) -> Any:
    evaluate_cfg = dict((pipeline_config or {}).get("evaluate") or {})
    if not bool(evaluate_cfg.get("enabled", True)):
        return {"metrics": {"skipped": "evaluation disabled by config"}}
    return PurityEvaluationStep(pipeline_config=pipeline_config).execute(
        payloads={"train": train_payload},
    )


@step(name="export_model", enable_cache=False)
def export_model_step(
    train_payload: Any,
    hpo_payload: Any = None,
    metrics: Any = None,
    pipeline_config: dict | None = None,
) -> Any:
    return PurityExportStep(pipeline_config=pipeline_config).execute(
        payloads={
            "train": train_payload,
            "hpo": hpo_payload,
            "metrics": metrics,
        }
    )


@pipeline
def training_pipeline(
    pipeline_config: dict | None = None,
):
    hpo_output = tune_model_step(pipeline_config=pipeline_config)
    train_output = train_model_step(hpo_payload=hpo_output, pipeline_config=pipeline_config)
    metrics = evaluate_model_step(train_payload=train_output, pipeline_config=pipeline_config)
    export = export_model_step(
        train_payload=train_output,
        hpo_payload=hpo_output,
        metrics=metrics,
        pipeline_config=pipeline_config,
    )
    return train_output, hpo_output, metrics, export

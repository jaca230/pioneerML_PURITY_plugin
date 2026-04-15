from __future__ import annotations

from pioneerml.pipeline.steps import BaseEvaluationStep


class PurityEvaluationStep(BaseEvaluationStep):
    step_key = "evaluate"

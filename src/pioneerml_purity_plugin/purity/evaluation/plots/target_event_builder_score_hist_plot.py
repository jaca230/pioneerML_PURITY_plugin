from __future__ import annotations

from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .target_diagnostics_shared import BinaryScoreHistogramBasePlot


@PLOT_REGISTRY_DEF.register("purity_event_builder_score_hist")
class PurityEventBuilderScoreHistPlot(BinaryScoreHistogramBasePlot):
    name = "purity_event_builder_score_hist"
    score_key = "event_scores"
    truth_key = "event_truth"
    title_default = "Event-Builder Token Score vs Truth"

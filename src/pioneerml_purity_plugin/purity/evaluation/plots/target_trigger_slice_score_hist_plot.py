from __future__ import annotations

from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .target_diagnostics_shared import BinaryScoreHistogramBasePlot


@PLOT_REGISTRY_DEF.register("purity_trigger_slice_score_hist")
class PurityTriggerSliceScoreHistPlot(BinaryScoreHistogramBasePlot):
    name = "purity_trigger_slice_score_hist"
    score_key = "trig_scores"
    truth_key = "trig_truth"
    title_default = "Trigger Slice Score vs Truth"

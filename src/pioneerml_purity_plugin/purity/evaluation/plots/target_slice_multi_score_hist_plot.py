from __future__ import annotations

from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .target_diagnostics_shared import BinaryScoreHistogramBasePlot


@PLOT_REGISTRY_DEF.register("purity_slice_multi_score_hist")
class PuritySliceMultiScoreHistPlot(BinaryScoreHistogramBasePlot):
    name = "purity_slice_multi_score_hist"
    score_key = "multi_scores"
    truth_key = "multi_truth"
    title_default = "Slice-Multi Score vs Truth"

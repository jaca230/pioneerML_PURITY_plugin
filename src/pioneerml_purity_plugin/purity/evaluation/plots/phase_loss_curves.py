from __future__ import annotations

from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .staged_loss_curves import _PuritySinglePhaseLossCurvesPlot


@PLOT_REGISTRY_DEF.register("purity_phase_1_loss_curves")
@PLOT_REGISTRY_DEF.register("purity_phase1_loss_curves")
class PurityPhase1LossCurvesPlot(_PuritySinglePhaseLossCurvesPlot):
    phase_index = 1
    name = "purity_phase_1_loss_curves"


@PLOT_REGISTRY_DEF.register("purity_phase_2_loss_curves")
@PLOT_REGISTRY_DEF.register("purity_phase2_loss_curves")
class PurityPhase2LossCurvesPlot(_PuritySinglePhaseLossCurvesPlot):
    phase_index = 2
    name = "purity_phase_2_loss_curves"


@PLOT_REGISTRY_DEF.register("purity_phase_3_loss_curves")
@PLOT_REGISTRY_DEF.register("purity_phase3_loss_curves")
class PurityPhase3LossCurvesPlot(_PuritySinglePhaseLossCurvesPlot):
    phase_index = 3
    name = "purity_phase_3_loss_curves"

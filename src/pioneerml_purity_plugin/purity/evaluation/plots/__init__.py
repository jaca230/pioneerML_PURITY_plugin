from __future__ import annotations

from typing import Any, Callable

from .staged_loss import (
    PurityPhase1LossCurvesPlot,
    PurityPhase2LossCurvesPlot,
    PurityPhase3LossCurvesPlot,
    PurityStagedLossCurvesPlot,
)


def _wrap(cls) -> Callable[..., Any]:
    def _fn(*args, **kwargs):
        return cls().render(*args, **kwargs)

    return _fn


plot_purity_staged_loss_curves = _wrap(PurityStagedLossCurvesPlot)
plot_purity_phase_1_loss_curves = _wrap(PurityPhase1LossCurvesPlot)
plot_purity_phase_2_loss_curves = _wrap(PurityPhase2LossCurvesPlot)
plot_purity_phase_3_loss_curves = _wrap(PurityPhase3LossCurvesPlot)

__all__ = [
    "PurityStagedLossCurvesPlot",
    "PurityPhase1LossCurvesPlot",
    "PurityPhase2LossCurvesPlot",
    "PurityPhase3LossCurvesPlot",
    "plot_purity_staged_loss_curves",
    "plot_purity_phase_1_loss_curves",
    "plot_purity_phase_2_loss_curves",
    "plot_purity_phase_3_loss_curves",
]


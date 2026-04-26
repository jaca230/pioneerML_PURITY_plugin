from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _extract_phase_histories, _plot_single_phase_axis, _save_and_show


@PLOT_REGISTRY_DEF.register("purity_staged_loss_curves")
class PurityStagedLossCurvesPlot(BasePlot):
    """Render one loss subplot per staged training phase."""

    name = "purity_staged_loss_curves"

    def render(
        self,
        *,
        module: object | None = None,
        phase_histories: Sequence[Mapping[str, Any]] | None = None,
        title: str = "PURITY Staged Loss Curves",
        log_scale: bool = True,
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        phases = _extract_phase_histories(module=module, phase_histories=phase_histories)
        if not phases:
            return None

        n_rows = len(phases)
        fig, axes = plt.subplots(n_rows, 1, figsize=(7.0, max(2.6, 2.8 * n_rows)), squeeze=False)
        fig.suptitle(title)
        for row_index, phase in enumerate(phases):
            phase_title = f"Phase {int(phase['index'])}: {phase['name']}"
            _plot_single_phase_axis(
                ax=axes[row_index][0],
                train_losses=phase["train_losses"],
                val_losses=phase["val_losses"],
                title=phase_title,
                log_scale=bool(log_scale),
            )
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)


class _PuritySinglePhaseLossCurvesPlot(BasePlot):
    phase_index: int = 1
    name = "purity_phase_loss_curves"

    def render(
        self,
        *,
        module: object | None = None,
        phase_histories: Sequence[Mapping[str, Any]] | None = None,
        title: str | None = None,
        log_scale: bool = True,
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        phases = _extract_phase_histories(module=module, phase_histories=phase_histories)
        idx = int(self.phase_index) - 1
        if idx < 0 or idx >= len(phases):
            return None
        phase = phases[idx]
        phase_title = title or f"PURITY Phase {int(phase['index'])}: {phase['name']}"

        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        _plot_single_phase_axis(
            ax=ax,
            train_losses=phase["train_losses"],
            val_losses=phase["val_losses"],
            title=phase_title,
            log_scale=bool(log_scale),
        )
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

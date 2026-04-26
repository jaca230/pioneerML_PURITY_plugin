from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _save_and_show
from .target_diagnostics_shared import resolve_target_diagnostics


@PLOT_REGISTRY_DEF.register("purity_positron_angle_error_hist")
class PurityPositronAngleErrorHistPlot(BasePlot):
    name = "purity_positron_angle_error_hist"

    def render(
        self,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        task_weights: Mapping[str, float] | None = None,
        max_batches: int = 200,
        bins: int = 120,
        title: str = "Positron Theta Error (truth - pred)",
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        diag = resolve_target_diagnostics(
            diagnostics=diagnostics,
            module=module,
            val_dataloader=val_dataloader,
            task_weights=task_weights,
            max_batches=max_batches,
        )
        if not diag:
            return None

        vals = np.asarray(diag.get("theta_diff_deg"), dtype=np.float32)
        if int(vals.size) <= 0:
            return None

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.hist(vals, bins=np.linspace(-120.0, 120.0, int(bins)))
        rms = float(np.sqrt(np.mean(vals ** 2)))
        mean = float(np.mean(vals))
        ax.set_title(f"{title}\nN={int(vals.size)}, RMS={rms:.2f} deg, Mean={mean:.2f} deg")
        ax.set_xlabel("deg")
        ax.set_ylabel("count")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

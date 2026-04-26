from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _save_and_show
from .target_diagnostics_shared import resolve_target_diagnostics


@PLOT_REGISTRY_DEF.register("purity_summary_energy_scatter")
class PuritySummaryEnergyScatterPlot(BasePlot):
    name = "purity_summary_energy_scatter"

    def render(
        self,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        task_weights: Mapping[str, float] | None = None,
        max_batches: int = 200,
        title: str = "Summary Positron Energy (raw units): Truth vs Pred",
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

        pred = np.asarray(diag.get("energy_pred_plot", diag.get("energy_pred")), dtype=np.float32)
        truth = np.asarray(diag.get("energy_truth_plot", diag.get("energy_truth")), dtype=np.float32)
        m = min(int(pred.size), int(truth.size))
        if m <= 0:
            return None
        pred = pred[:m]
        truth = truth[:m]
        valid = np.isfinite(pred) & np.isfinite(truth)
        pred = pred[valid]
        truth = truth[valid]
        if int(pred.size) <= 0:
            return None

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.scatter(truth, pred, s=8, alpha=0.35)
        lo = float(np.nanmin([truth.min(), pred.min()]))
        hi = float(np.nanmax([truth.max(), pred.max()]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ttl = f"{title}\nN={int(pred.size)}"
        ax.set_title(ttl)
        ax.set_xlabel("truth")
        ax.set_ylabel("pred")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _save_and_show
from .target_diagnostics_shared import resolve_target_diagnostics


@PLOT_REGISTRY_DEF.register("purity_positron_theta_scatter")
class PurityPositronThetaScatterPlot(BasePlot):
    name = "purity_positron_theta_scatter"

    def render(
        self,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        task_weights: Mapping[str, float] | None = None,
        max_batches: int = 200,
        title: str = "Positron Theta: Truth vs Pred",
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

        pred = np.asarray(diag.get("theta_pred_deg"), dtype=np.float32)
        truth = np.asarray(diag.get("theta_truth_deg"), dtype=np.float32)
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

        stats = diag.get("sample_stats") if isinstance(diag.get("sample_stats"), Mapping) else {}
        n_cand = int(stats.get("theta_graph_candidates", 0))
        n_used = int(pred.size)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.scatter(truth, pred, s=8, alpha=0.35)
        lo = float(np.nanmin([truth.min(), pred.min()]))
        hi = float(np.nanmax([truth.max(), pred.max()]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(0.0, 180.0)
        ax.set_ylim(0.0, 180.0)
        ax.set_title(f"{title}\nN={n_used}, trigger-positron candidates={n_cand}")
        ax.set_xlabel("truth theta (deg)")
        ax.set_ylabel("pred theta (deg)")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _save_and_show
from .target_diagnostics_shared import resolve_target_diagnostics


@PLOT_REGISTRY_DEF.register("purity_summary_time_scatter")
class PuritySummaryTimeScatterPlot(BasePlot):
    name = "purity_summary_time_scatter"

    def render(
        self,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        task_weights: Mapping[str, float] | None = None,
        max_batches: int = 200,
        title: str = "Summary Positron Time: Proxy Truth vs Pred",
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

        stats = diag.get("sample_stats") if isinstance(diag.get("sample_stats"), Mapping) else {}
        n_removed = int(stats.get("time_pairs_sentinel_removed", 0))
        n_finite = int(stats.get("time_pairs_finite", 0))
        frac_removed = float(stats.get("time_pairs_sentinel_fraction_of_finite", 0.0))

        pred = np.asarray(diag.get("time_pred_filtered", diag.get("time_pred")), dtype=np.float32)
        truth = np.asarray(diag.get("time_truth_proxy_filtered", diag.get("time_truth_proxy")), dtype=np.float32)
        m = min(int(pred.size), int(truth.size))
        if m > 0:
            pred = pred[:m]
            truth = truth[:m]
            valid = np.isfinite(pred) & np.isfinite(truth)
            pred = pred[valid]
            truth = truth[valid]
        else:
            pred = np.asarray([], dtype=np.float32)
            truth = np.asarray([], dtype=np.float32)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        if int(pred.size) > 0:
            ax.scatter(truth, pred, s=8, alpha=0.35)
            lo = float(np.nanmin([truth.min(), pred.min()]))
            hi = float(np.nanmax([truth.max(), pred.max()]))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        else:
            ax.text(
                0.5,
                0.5,
                "No non-sentinel finite time pairs after filtering",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title(
            f"{title}\nN={int(pred.size)}, dropped_sentinel={n_removed}/{n_finite} ({100.0 * frac_removed:.1f}%)"
        )
        ax.set_xlabel("truth proxy")
        ax.set_ylabel("pred")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

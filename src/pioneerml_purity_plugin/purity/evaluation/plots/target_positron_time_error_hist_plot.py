from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _save_and_show
from .target_diagnostics_shared import resolve_target_diagnostics


@PLOT_REGISTRY_DEF.register("purity_positron_time_error_hist")
class PurityPositronTimeErrorHistPlot(BasePlot):
    name = "purity_positron_time_error_hist"

    def render(
        self,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        task_weights: Mapping[str, float] | None = None,
        max_batches: int = 200,
        bins: int = 120,
        title: str = "Positron Time Error (proxy truth - pred)",
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
            vals = (truth[:m] - pred[:m]).astype(np.float32, copy=False)
            vals = vals[np.isfinite(vals)]
        else:
            vals = np.asarray([], dtype=np.float32)

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        if int(vals.size) > 0:
            ax.hist(vals, bins=int(bins), alpha=0.8)
            rms = float(np.sqrt(np.mean(vals ** 2)))
            mean = float(np.mean(vals))
            extra = f"N={int(vals.size)}, RMS={rms:.3f}, Mean={mean:.3f}, "
        else:
            ax.text(
                0.5,
                0.5,
                "No non-sentinel finite time pairs after filtering",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            extra = "N=0, "

        ax.set_title(
            f"{title}\n{extra}dropped_sentinel={n_removed}/{n_finite} ({100.0 * frac_removed:.1f}%)"
        )
        ax.set_xlabel("time error")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _save_and_show
from .target_diagnostics_shared import resolve_target_diagnostics


@PLOT_REGISTRY_DEF.register("purity_pion_stop_error_hist")
class PurityPionStopErrorHistPlot(BasePlot):
    name = "purity_pion_stop_error_hist"

    def render(
        self,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        task_weights: Mapping[str, float] | None = None,
        max_batches: int = 200,
        bins: int = 60,
        title: str = "Pion Stop Error (mm)",
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

        eu = np.asarray(diag.get("pion_euclid"), dtype=np.float32)
        dx = np.asarray(diag.get("pion_dx"), dtype=np.float32)
        dy = np.asarray(diag.get("pion_dy"), dtype=np.float32)
        dz = np.asarray(diag.get("pion_dz"), dtype=np.float32)
        if int(eu.size) + int(dx.size) + int(dy.size) + int(dz.size) <= 0:
            return None

        stats = diag.get("sample_stats") if isinstance(diag.get("sample_stats"), Mapping) else {}
        n_total = int(stats.get("pion_graph_pairs_total", 0))
        n_valid = int(stats.get("pion_graph_pairs_valid_truth_mask", 0))
        n_used = int(stats.get("pion_graph_pairs_used", int(eu.size)))

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
        fig.suptitle(
            f"{title}\nN={n_used}, valid_truth={n_valid}, total_graphs={n_total} "
            "(valid_truth=finite truth_pion_stop_x/y/z and graph-level target present)"
        )
        series = [
            ("euclidean", eu),
            ("x_error", dx),
            ("y_error", dy),
            ("z_error", dz),
        ]
        for ax, (label, values) in zip(axes.reshape(-1), series, strict=False):
            if int(values.size) > 0:
                ax.hist(values, bins=int(bins), alpha=0.8)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label} (N={int(values.size)})")
            ax.set_xlabel("mm")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

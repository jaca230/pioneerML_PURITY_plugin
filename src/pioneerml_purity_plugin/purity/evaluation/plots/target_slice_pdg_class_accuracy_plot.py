from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _save_and_show
from .target_diagnostics_shared import CLASS_NAMES, resolve_target_diagnostics


@PLOT_REGISTRY_DEF.register("purity_slice_pdg_class_accuracy")
class PuritySlicePDGClassAccuracyPlot(BasePlot):
    name = "purity_slice_pdg_class_accuracy"

    def render(
        self,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        task_weights: Mapping[str, float] | None = None,
        max_batches: int = 200,
        title: str = "Slice PDG Confusion Matrix",
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
        conf = np.asarray(diag.get("slice_conf"), dtype=np.int64)
        if conf.shape != (3, 3):
            return None

        row_sum = conf.sum(axis=1, keepdims=True).clip(min=1)
        conf_norm = conf.astype(np.float32) / row_sum.astype(np.float32)

        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(conf_norm, cmap="Greens", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Truth class")
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

        for i in range(conf.shape[0]):
            for j in range(conf.shape[1]):
                frac = float(conf_norm[i, j])
                cnt = int(conf[i, j])
                color = "white" if frac > 0.5 else "black"
                ax.text(j, i, f"{frac:.2f}\n({cnt})", ha="center", va="center", color=color, fontsize=9)

        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

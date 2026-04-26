from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _extract_pdg_batch_histories, _extract_pdg_phase_regions, _save_and_show


@PLOT_REGISTRY_DEF.register("purity_pdg_accuracy_curves")
class PurityPDGAccuracyCurvesPlot(BasePlot):
    """Render per-batch node-PDG accuracy curves (overall + per class)."""

    name = "purity_pdg_accuracy_curves"

    def render(
        self,
        *,
        module: object | None = None,
        split: str = "train",
        pdg_histories: Sequence[Mapping[str, Any]] | None = None,
        title: str = "PURITY Node-PDG Accuracy (Per Batch)",
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        records = _extract_pdg_batch_histories(module=module, split=split, histories=pdg_histories)
        if not records:
            return None

        phase_regions = _extract_pdg_phase_regions(module=module, split=split)
        x = [int(item["global_batch_index"]) for item in records]
        overall = [float(item["overall_accuracy"]) for item in records]

        class_names: list[str] = []
        for item in records:
            for name in dict(item.get("class_accuracy") or {}).keys():
                if name not in class_names:
                    class_names.append(str(name))

        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        region_colors = ["#d9edf7", "#dff0d8", "#fcf8e3", "#f2dede", "#e9e7fd"]
        y_max = 1.02
        for idx, region in enumerate(phase_regions):
            start = int(region["start"])
            end = int(region["end"])
            color = region_colors[idx % len(region_colors)]
            ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=0.24, lw=0)
            center = (start + end - 1) / 2.0
            ax.text(center, y_max - 0.01, str(region["name"]), ha="center", va="top", fontsize=8)

        ax.plot(x, overall, label="overall", color="black", linewidth=2.0)
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        for idx, class_name in enumerate(class_names):
            y = [float(dict(item.get("class_accuracy") or {}).get(class_name, float("nan"))) for item in records]
            ax.plot(x, y, label=class_name, color=palette[idx % len(palette)], linewidth=1.5, alpha=0.95)

        ax.set_ylim(0.0, y_max)
        ax.set_xlim(-0.5, max(x) + 0.5)
        ax.set_xlabel("Batch Index")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title} [{str(split)}]")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right")
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

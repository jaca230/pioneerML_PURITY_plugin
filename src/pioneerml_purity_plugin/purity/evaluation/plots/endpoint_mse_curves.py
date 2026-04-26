from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import (
    _extract_endpoint_batch_histories,
    _extract_endpoint_class_mse_names,
    _extract_endpoint_phase_regions,
    _save_and_show,
)


@PLOT_REGISTRY_DEF.register("purity_endpoint_mse_curves")
class PurityEndpointMSECurvesPlot(BasePlot):
    """Render per-batch endpoint MSE curves (overall/start/stop/x/y/z)."""

    name = "purity_endpoint_mse_curves"

    def render(
        self,
        *,
        module: object | None = None,
        split: str = "train",
        endpoint_histories: Sequence[Mapping[str, Any]] | None = None,
        title: str = "PURITY Endpoint MSE (Per Batch)",
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        records = _extract_endpoint_batch_histories(module=module, split=split, histories=endpoint_histories)
        if not records:
            return None

        phase_regions = _extract_endpoint_phase_regions(module=module, split=split)
        x = [int(item["global_batch_index"]) for item in records]
        overall = [float(item["overall_mse"]) for item in records]
        start = [float(item["start_mse"]) for item in records]
        stop = [float(item["stop_mse"]) for item in records]
        x_axis = [float(dict(item.get("axis_mse") or {}).get("x", float("nan"))) for item in records]
        y_axis = [float(dict(item.get("axis_mse") or {}).get("y", float("nan"))) for item in records]
        z_axis = [float(dict(item.get("axis_mse") or {}).get("z", float("nan"))) for item in records]

        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        region_colors = ["#d9edf7", "#dff0d8", "#fcf8e3", "#f2dede", "#e9e7fd"]
        y_max = max(overall + start + stop + [0.0])
        y_max = max(1e-8, float(y_max)) * 1.05
        for idx, region in enumerate(phase_regions):
            start_b = int(region["start"])
            end_b = int(region["end"])
            color = region_colors[idx % len(region_colors)]
            ax.axvspan(start_b - 0.5, end_b - 0.5, color=color, alpha=0.24, lw=0)
            center = (start_b + end_b - 1) / 2.0
            ax.text(center, y_max * 0.98, str(region["name"]), ha="center", va="top", fontsize=8)

        ax.plot(x, overall, label="overall_mse", color="black", linewidth=2.0)
        ax.plot(x, start, label="start_mse", color="tab:blue", linewidth=1.5, alpha=0.95)
        ax.plot(x, stop, label="stop_mse", color="tab:orange", linewidth=1.5, alpha=0.95)
        ax.plot(x, x_axis, label="x_mse", color="tab:green", linewidth=1.2, alpha=0.9)
        ax.plot(x, y_axis, label="y_mse", color="tab:red", linewidth=1.2, alpha=0.9)
        ax.plot(x, z_axis, label="z_mse", color="tab:purple", linewidth=1.2, alpha=0.9)

        ax.set_xlim(-0.5, max(x) + 0.5)
        if all(math.isfinite(float(v)) and float(v) > 0.0 for v in overall if v is not None):
            ax.set_yscale("log")
        ax.set_xlabel("Batch Index")
        ax.set_ylabel("MSE")
        ax.set_title(f"{title} [{str(split)}]")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)


@PLOT_REGISTRY_DEF.register("purity_endpoint_mse_by_particle_curves")
class PurityEndpointMSEByParticleCurvesPlot(BasePlot):
    """Render per-batch endpoint MSE curves split by truth particle type."""

    name = "purity_endpoint_mse_by_particle_curves"

    def render(
        self,
        *,
        module: object | None = None,
        split: str = "train",
        endpoint_histories: Sequence[Mapping[str, Any]] | None = None,
        title: str = "PURITY Endpoint MSE by Truth Particle (Per Batch)",
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        records = _extract_endpoint_batch_histories(module=module, split=split, histories=endpoint_histories)
        if not records:
            return None

        class_names = _extract_endpoint_class_mse_names(records)
        if not class_names:
            return None

        phase_regions = _extract_endpoint_phase_regions(module=module, split=split)
        x = [int(item["global_batch_index"]) for item in records]
        overall = [float(item["overall_mse"]) for item in records]

        fig, ax = plt.subplots(figsize=(9.0, 4.0))
        region_colors = ["#d9edf7", "#dff0d8", "#fcf8e3", "#f2dede", "#e9e7fd"]
        y_max = max(overall + [0.0])
        for cls in class_names:
            vals = [float(dict(item.get("class_mse") or {}).get(cls, float("nan"))) for item in records]
            finite_vals = [v for v in vals if math.isfinite(float(v))]
            if finite_vals:
                y_max = max(y_max, max(finite_vals))
        y_max = max(1e-8, float(y_max)) * 1.05

        for idx, region in enumerate(phase_regions):
            start_b = int(region["start"])
            end_b = int(region["end"])
            color = region_colors[idx % len(region_colors)]
            ax.axvspan(start_b - 0.5, end_b - 0.5, color=color, alpha=0.24, lw=0)
            center = (start_b + end_b - 1) / 2.0
            ax.text(center, y_max * 0.98, str(region["name"]), ha="center", va="top", fontsize=8)

        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        for idx, cls in enumerate(class_names):
            vals = [float(dict(item.get("class_mse") or {}).get(cls, float("nan"))) for item in records]
            ax.plot(x, vals, label=f"{cls}_mse", color=palette[idx % len(palette)], linewidth=1.7, alpha=0.95)

        ax.plot(x, overall, label="overall_mse", color="black", linewidth=1.4, linestyle="--", alpha=0.85)

        ax.set_xlim(-0.5, max(x) + 0.5)
        if all(math.isfinite(float(v)) and float(v) > 0.0 for v in overall if v is not None):
            ax.set_yscale("log")
        ax.set_xlabel("Batch Index")
        ax.set_ylabel("MSE")
        ax.set_title(f"{title} [{str(split)}]")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

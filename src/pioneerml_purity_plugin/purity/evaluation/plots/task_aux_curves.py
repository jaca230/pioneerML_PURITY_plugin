from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _extract_task_diagnostic_batch_histories, _extract_task_phase_regions, _save_and_show


def _extract_aux_keys(records: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in records:
        losses = dict(item.get("losses") or {})
        for key in losses.keys():
            name = str(key)
            if name not in names:
                names.append(name)

    preferred_order = [
        "end_PosLoss",
        "end_SpanLoss",
        "end_DirLoss",
        "end_MeanWidth",
        "end_MeanError",
        "lyso_beta",
        "lyso_potential",
        "lyso_fraction",
    ]
    out: list[str] = [k for k in preferred_order if k in names]
    for name in names:
        if name not in out and (name.startswith("end_") or name.startswith("lyso_")):
            out.append(name)
    return out


def _can_log(values: Sequence[float]) -> bool:
    finite = [float(v) for v in values if isinstance(v, (float, int)) and math.isfinite(float(v))]
    return bool(finite) and all(v > 0.0 for v in finite)


def _pretty_label(key: str) -> str:
    mapping = {
        "end_PosLoss": "Endpoint Position Loss (internal)",
        "end_SpanLoss": "Endpoint Span Loss (internal)",
        "end_DirLoss": "Endpoint Direction Loss (internal)",
        "end_MeanWidth": "Endpoint Mean Width (internal)",
        "end_MeanError": "Endpoint Mean Error (internal)",
        "lyso_beta": "LYSO Condensation Beta (internal)",
        "lyso_potential": "LYSO Condensation Potential (internal)",
        "lyso_fraction": "LYSO Condensation Fraction (internal)",
    }
    return mapping.get(str(key), str(key))


@PLOT_REGISTRY_DEF.register("purity_task_aux_curves")
class PurityTaskAuxCurvesPlot(BasePlot):
    """Render per-batch auxiliary/internal terms (separate from weighted task losses)."""

    name = "purity_task_aux_curves"

    def render(
        self,
        *,
        module: object | None = None,
        split: str = "train",
        task_histories: Sequence[Mapping[str, Any]] | None = None,
        aux_keys: Sequence[str] | None = None,
        title: str = "PURITY Auxiliary/Internal Terms (Per Batch)",
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        records = _extract_task_diagnostic_batch_histories(module=module, split=split, histories=task_histories)
        if not records:
            return None

        keys = [str(k) for k in list(aux_keys or []) if str(k).strip() != ""]
        if not keys:
            keys = _extract_aux_keys(records)
        if not keys:
            return None

        x = [int(item["global_batch_index"]) for item in records]
        phase_regions = _extract_task_phase_regions(module=module, split=split)
        n = len(keys)
        n_cols = 2 if n > 6 else 1
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(8.5 * n_cols, max(2.6, 2.4 * n_rows)),
            squeeze=False,
        )
        fig.suptitle(f"{title} [{str(split)}]")
        region_colors = ["#d9edf7", "#dff0d8", "#fcf8e3", "#f2dede", "#e9e7fd"]

        for idx, key in enumerate(keys):
            ax = axes[idx // n_cols][idx % n_cols]
            y = [float(dict(item.get("losses") or {}).get(key, float("nan"))) for item in records]
            for ridx, region in enumerate(phase_regions):
                start = int(region["start"])
                end = int(region["end"])
                ax.axvspan(start - 0.5, end - 0.5, color=region_colors[ridx % len(region_colors)], alpha=0.22, lw=0)
            ax.plot(x, y, linewidth=1.4, color="tab:purple")
            ax.set_title(_pretty_label(key))
            ax.set_xlabel("Batch Index")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.2)
            if x:
                ax.set_xlim(-0.5, max(x) + 0.5)
            if _can_log(y):
                ax.set_yscale("log")

        for idx in range(n, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].axis("off")

        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

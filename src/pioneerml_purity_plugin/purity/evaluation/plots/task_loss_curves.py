from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _extract_task_diagnostic_batch_histories, _extract_task_phase_regions, _save_and_show


def _extract_loss_keys(records: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in records:
        losses = dict(item.get("losses") or {})
        for key in losses.keys():
            name = str(key)
            if name not in names:
                names.append(name)

    weighted_order = [
        "loss_total",
        "loss_node_pdg",
        "loss_slice_pdg",
        "loss_slice_multi",
        "loss_trigger_slice",
        "loss_atar_edge",
        "loss_pion_kinematics",
        "loss_positron_angle",
        "loss_lyso_condensation",
        "L_event_builder",
    ]
    # Strictly keep weighted task objectives in this plot.
    return [k for k in weighted_order if k in names]


def _can_log(values: Sequence[float]) -> bool:
    finite = [float(v) for v in values if isinstance(v, (float, int)) and math.isfinite(float(v))]
    return bool(finite) and all(v > 0.0 for v in finite)


def _weight_key_for_loss(loss_key: str) -> str | None:
    mapping = {
        "loss_node_pdg": "w_node_pdg",
        "loss_slice_pdg": "w_slice_pdg",
        "loss_slice_multi": "w_atar_slice_multi",
        "loss_trigger_slice": "w_atar_trigger_slice",
        "loss_atar_edge": "w_endpoints",
        "loss_pion_kinematics": "w_pion_kinematics",
        "loss_positron_angle": "w_positron_angle",
        "loss_lyso_condensation": "w_lyso_condensation",
        "L_event_builder": "w_event_builder",
    }
    return mapping.get(str(loss_key))


def _weight_value_for_loss(module: object | None, loss_key: str) -> float | None:
    if module is None:
        return None
    weights = getattr(module, "_task_weights", None)
    if not isinstance(weights, Mapping):
        return None
    w_key = _weight_key_for_loss(loss_key)
    if not w_key:
        return None
    value = weights.get(w_key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@PLOT_REGISTRY_DEF.register("purity_task_loss_curves")
class PurityTaskLossCurvesPlot(BasePlot):
    """Render per-batch task-loss curves (one subplot per loss term)."""

    name = "purity_task_loss_curves"

    def render(
        self,
        *,
        module: object | None = None,
        split: str = "train",
        task_histories: Sequence[Mapping[str, Any]] | None = None,
        loss_keys: Sequence[str] | None = None,
        title: str = "PURITY Task Losses (Per Batch)",
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        records = _extract_task_diagnostic_batch_histories(module=module, split=split, histories=task_histories)
        if not records:
            return None

        keys = [str(k) for k in list(loss_keys or []) if str(k).strip() != ""]
        if not keys:
            keys = _extract_loss_keys(records)
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
            ax.plot(x, y, linewidth=1.4, color="tab:blue")
            w = _weight_value_for_loss(module=module, loss_key=key)
            if w is not None:
                ax.set_title(f"{key} (w={w:g})")
            else:
                ax.set_title(str(key))
            ax.set_xlabel("Batch Index")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.2)
            if x:
                ax.set_xlim(-0.5, max(x) + 0.5)
            if _can_log(y):
                ax.set_yscale("log")

        for idx in range(n, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].axis("off")

        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover - optional
    display = None


def _to_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        try:
            values = values.detach().cpu().tolist()
        except Exception:
            pass
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        out: list[float] = []
        for value in values:
            try:
                out.append(float(value))
            except Exception:
                continue
        return out
    return []


def _align_histories(train_losses: list[float], val_losses: list[float]) -> tuple[list[float], list[float]]:
    train_hist = list(train_losses)
    val_hist = list(val_losses)
    # Lightning can prepend sanity-val points. Keep alignment consistent with
    # core `loss_curves` behavior.
    while len(val_hist) > len(train_hist) and len(train_hist) > 0:
        val_hist = val_hist[1:]
    return train_hist, val_hist


def _extract_phase_histories(
    *,
    module: object | None = None,
    phase_histories: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw = phase_histories
    if raw is None and module is not None:
        raw = getattr(module, "staged_phase_loss_histories", None)

    out: list[dict[str, Any]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or f"phase_{index}").strip() or f"phase_{index}"
            train_hist = _to_float_list(item.get("train_losses"))
            val_hist = _to_float_list(item.get("val_losses"))
            train_hist, val_hist = _align_histories(train_hist, val_hist)
            out.append(
                {
                    "index": int(item.get("index") or index),
                    "name": name,
                    "train_losses": train_hist,
                    "val_losses": val_hist,
                }
            )

    if out:
        return out

    # Fallback when staged history is unavailable.
    if module is not None:
        train_hist = _to_float_list(getattr(module, "train_epoch_loss_history", []))
        val_hist = _to_float_list(getattr(module, "val_epoch_loss_history", []))
        train_hist, val_hist = _align_histories(train_hist, val_hist)
        if train_hist or val_hist:
            return [
                {
                    "index": 1,
                    "name": "training",
                    "train_losses": train_hist,
                    "val_losses": val_hist,
                }
            ]
    return []


def _can_use_log_scale(values: Sequence[float]) -> bool:
    return bool(values) and all(math.isfinite(float(v)) and float(v) > 0.0 for v in values)


def _extract_pdg_batch_histories(
    *,
    module: object | None = None,
    split: str = "train",
    histories: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw = histories
    if raw is None and module is not None:
        if str(split) == "val":
            raw = getattr(module, "val_pdg_batch_accuracy_history", None)
        else:
            raw = getattr(module, "train_pdg_batch_accuracy_history", None)

    out: list[dict[str, Any]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            class_accuracy_raw = item.get("class_accuracy")
            if not isinstance(class_accuracy_raw, Mapping):
                class_accuracy_raw = {}
            class_accuracy = {
                str(key): float(value)
                for key, value in dict(class_accuracy_raw).items()
                if isinstance(value, (int, float))
            }
            if not class_accuracy and item.get("overall_accuracy") is None:
                continue
            out.append(
                {
                    "global_batch_index": int(item.get("global_batch_index", index)),
                    "overall_accuracy": float(item.get("overall_accuracy", 0.0)),
                    "class_accuracy": class_accuracy,
                    "phase_index": item.get("phase_index"),
                    "phase_name": item.get("phase_name"),
                }
            )
    return out


def _extract_pdg_phase_regions(
    *,
    module: object | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    raw = getattr(module, "staged_phase_pdg_histories", None) if module is not None else None
    out: list[dict[str, Any]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        cursor = 0
        for idx, item in enumerate(raw, start=1):
            if not isinstance(item, Mapping):
                continue
            if str(split) == "val":
                count = int(item.get("val_batches", 0))
            else:
                count = int(item.get("train_batches", 0))
            if count <= 0:
                continue
            start = int(cursor)
            end = int(cursor + count)
            cursor = end
            out.append(
                {
                    "index": int(item.get("index", idx)),
                    "name": str(item.get("name") or f"phase_{idx}"),
                    "start": start,
                    "end": end,
                }
            )
    return out


def _plot_single_phase_axis(
    *,
    ax,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    title: str,
    log_scale: bool,
) -> None:
    train_hist = list(train_losses)
    val_hist = list(val_losses)
    ax_secondary = None
    if train_hist and val_hist:
        ax.plot(train_hist, color="tab:blue", label="train_loss")
        ax_secondary = ax.twinx()
        ax_secondary.plot(val_hist, color="tab:orange", label="val_loss")
        ax.set_ylabel("Train Loss")
        ax_secondary.set_ylabel("Validation Loss")
    else:
        if train_hist:
            ax.plot(train_hist, color="tab:blue", label="train_loss")
        if val_hist:
            ax.plot(val_hist, color="tab:orange", label="val_loss")
        ax.set_ylabel("Loss")

    if log_scale:
        if _can_use_log_scale(train_hist):
            ax.set_yscale("log")
        if ax_secondary is not None:
            if _can_use_log_scale(val_hist):
                ax_secondary.set_yscale("log")
        elif _can_use_log_scale(val_hist):
            ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Epoch")

    handles, labels = ax.get_legend_handles_labels()
    if ax_secondary is not None:
        h2, l2 = ax_secondary.get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax.legend(handles, labels)


def _save_and_show(*, fig, save_path: str | None, show: bool) -> str | None:
    if save_path is not None:
        path = Path(str(save_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
    if show:
        backend = plt.get_backend().lower()
        if backend.startswith("agg"):
            if display is not None:
                try:
                    display(fig)
                except Exception:
                    pass
        else:
            plt.show()
    plt.close(fig)
    return save_path


@PLOT_REGISTRY_DEF.register("purity_staged_loss_curves")
class PurityStagedLossCurvesPlot(BasePlot):
    """Render one loss subplot per staged training phase."""

    name = "purity_staged_loss_curves"

    def render(
        self,
        *,
        module: object | None = None,
        phase_histories: Sequence[Mapping[str, Any]] | None = None,
        title: str = "PURITY Staged Loss Curves",
        log_scale: bool = True,
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        phases = _extract_phase_histories(module=module, phase_histories=phase_histories)
        if not phases:
            return None

        n_rows = len(phases)
        fig, axes = plt.subplots(n_rows, 1, figsize=(7.0, max(2.6, 2.8 * n_rows)), squeeze=False)
        fig.suptitle(title)
        for row_index, phase in enumerate(phases):
            phase_title = f"Phase {int(phase['index'])}: {phase['name']}"
            _plot_single_phase_axis(
                ax=axes[row_index][0],
                train_losses=phase["train_losses"],
                val_losses=phase["val_losses"],
                title=phase_title,
                log_scale=bool(log_scale),
            )
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)


class _PuritySinglePhaseLossCurvesPlot(BasePlot):
    phase_index: int = 1
    name = "purity_phase_loss_curves"

    def render(
        self,
        *,
        module: object | None = None,
        phase_histories: Sequence[Mapping[str, Any]] | None = None,
        title: str | None = None,
        log_scale: bool = True,
        save_path: str | None = None,
        show: bool = False,
    ) -> str | None:
        phases = _extract_phase_histories(module=module, phase_histories=phase_histories)
        idx = int(self.phase_index) - 1
        if idx < 0 or idx >= len(phases):
            return None
        phase = phases[idx]
        phase_title = title or f"PURITY Phase {int(phase['index'])}: {phase['name']}"

        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        _plot_single_phase_axis(
            ax=ax,
            train_losses=phase["train_losses"],
            val_losses=phase["val_losses"],
            title=phase_title,
            log_scale=bool(log_scale),
        )
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)


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
        records = _extract_pdg_batch_histories(
            module=module,
            split=split,
            histories=pdg_histories,
        )
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


@PLOT_REGISTRY_DEF.register("purity_phase_1_loss_curves")
@PLOT_REGISTRY_DEF.register("purity_phase1_loss_curves")
class PurityPhase1LossCurvesPlot(_PuritySinglePhaseLossCurvesPlot):
    phase_index = 1
    name = "purity_phase_1_loss_curves"


@PLOT_REGISTRY_DEF.register("purity_phase_2_loss_curves")
@PLOT_REGISTRY_DEF.register("purity_phase2_loss_curves")
class PurityPhase2LossCurvesPlot(_PuritySinglePhaseLossCurvesPlot):
    phase_index = 2
    name = "purity_phase_2_loss_curves"


@PLOT_REGISTRY_DEF.register("purity_phase_3_loss_curves")
@PLOT_REGISTRY_DEF.register("purity_phase3_loss_curves")
class PurityPhase3LossCurvesPlot(_PuritySinglePhaseLossCurvesPlot):
    phase_index = 3
    name = "purity_phase_3_loss_curves"

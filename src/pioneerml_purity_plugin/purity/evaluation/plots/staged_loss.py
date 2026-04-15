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

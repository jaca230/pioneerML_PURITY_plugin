from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

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
    elif hasattr(values, "tolist"):
        try:
            values = values.tolist()
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


def _extract_endpoint_batch_histories(
    *,
    module: object | None = None,
    split: str = "train",
    histories: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw = histories
    if raw is None and module is not None:
        if str(split) == "val":
            raw = getattr(module, "val_endpoint_mse_batch_history", None)
        else:
            raw = getattr(module, "train_endpoint_mse_batch_history", None)

    out: list[dict[str, Any]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            axis_raw = item.get("axis_mse")
            axis = dict(axis_raw) if isinstance(axis_raw, Mapping) else {}
            class_raw = item.get("class_mse")
            class_mse = dict(class_raw) if isinstance(class_raw, Mapping) else {}
            counts_raw = item.get("class_counts")
            class_counts = dict(counts_raw) if isinstance(counts_raw, Mapping) else {}
            out.append(
                {
                    "global_batch_index": int(item.get("global_batch_index", index)),
                    "overall_mse": float(item.get("overall_mse", 0.0)),
                    "start_mse": float(item.get("start_mse", 0.0)),
                    "stop_mse": float(item.get("stop_mse", 0.0)),
                    "axis_mse": {
                        "x": float(axis.get("x", float("nan"))),
                        "y": float(axis.get("y", float("nan"))),
                        "z": float(axis.get("z", float("nan"))),
                    },
                    "class_mse": {
                        str(k): float(v)
                        for k, v in class_mse.items()
                        if isinstance(v, (int, float))
                    },
                    "class_counts": {
                        str(k): int(v)
                        for k, v in class_counts.items()
                        if isinstance(v, (int, float))
                    },
                    "phase_index": item.get("phase_index"),
                    "phase_name": item.get("phase_name"),
                }
            )
    return out


def _extract_endpoint_phase_regions(
    *,
    module: object | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    raw = getattr(module, "staged_phase_endpoint_histories", None) if module is not None else None
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


def _extract_endpoint_class_mse_names(records: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in records:
        class_raw = item.get("class_mse")
        class_map = dict(class_raw) if isinstance(class_raw, Mapping) else {}
        for key in class_map.keys():
            name = str(key)
            if name not in names:
                names.append(name)
    return names


def _extract_task_diagnostic_batch_histories(
    *,
    module: object | None = None,
    split: str = "train",
    histories: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    raw = histories
    if raw is None and module is not None:
        if str(split) == "val":
            raw = getattr(module, "val_task_diagnostics_batch_history", None)
        else:
            raw = getattr(module, "train_task_diagnostics_batch_history", None)

    out: list[dict[str, Any]] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            losses_raw = item.get("losses")
            losses = dict(losses_raw) if isinstance(losses_raw, Mapping) else {}
            metrics_raw = item.get("metrics")
            metrics = dict(metrics_raw) if isinstance(metrics_raw, Mapping) else {}
            out.append(
                {
                    "global_batch_index": int(item.get("global_batch_index", index)),
                    "losses": {
                        str(k): float(v)
                        for k, v in losses.items()
                        if isinstance(v, (int, float))
                    },
                    "metrics": {
                        str(k): float(v)
                        for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    },
                    "phase_index": item.get("phase_index"),
                    "phase_name": item.get("phase_name"),
                }
            )
    return out


def _extract_task_phase_regions(
    *,
    module: object | None = None,
    split: str = "train",
) -> list[dict[str, Any]]:
    raw = getattr(module, "staged_phase_task_histories", None) if module is not None else None
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

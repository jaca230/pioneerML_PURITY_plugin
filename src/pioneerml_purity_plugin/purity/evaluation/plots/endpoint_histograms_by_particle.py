from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from pioneerml.evaluation.plots.base_plot import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY_DEF

from .common import _save_and_show, _to_float_list


def _collect_endpoint_pred_truth_maps(
    *,
    module: Any,
    val_dataloader: Any,
    class_names: Sequence[str],
    coord_names: Sequence[str],
    max_batches: int | None = 100,
    task_weights: Mapping[str, float] | None = None,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    from pioneerml_purity_plugin.purity.losses.utils.unified_training_components import format_targets_from_batch

    pred_collected = {
        str(cls): {str(coord): [] for coord in coord_names}
        for cls in class_names
    }
    truth_collected = {
        str(cls): {str(coord): [] for coord in coord_names}
        for cls in class_names
    }

    if module is None or val_dataloader is None:
        pred_map = {
            str(cls): {
                str(coord): np.array([], dtype=np.float32)
                for coord in coord_names
            }
            for cls in class_names
        }
        truth_map = {
            str(cls): {
                str(coord): np.array([], dtype=np.float32)
                for coord in coord_names
            }
            for cls in class_names
        }
        pred_counts = {
            str(cls): {str(coord): 0 for coord in coord_names}
            for cls in class_names
        }
        truth_counts = {
            str(cls): {str(coord): 0 for coord in coord_names}
            for cls in class_names
        }
        return pred_map, truth_map, pred_counts, truth_counts

    device = next(module.parameters()).device
    model = module.model.eval()
    active_task_weights = dict(task_weights) if isinstance(task_weights, Mapping) else getattr(module, "_task_weights", None)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batch = batch.to(device)
            if active_task_weights:
                raw_preds = model(batch, task_weights=active_task_weights)
            else:
                raw_preds = model(batch)

            if not isinstance(raw_preds, Mapping) or "atar_endpoints" not in raw_preds:
                continue
            preds = raw_preds["atar_endpoints"]
            if not isinstance(preds, torch.Tensor) or preds.dim() != 4 or int(preds.size(-1)) < 2:
                continue

            target = getattr(batch, "atar_slice_pdg_target", None)
            if not isinstance(target, torch.Tensor) or target.dim() != 2:
                continue
            target = target.to(device=preds.device, dtype=torch.float32)

            pred_xyz = preds[..., 1][:, :, :3] * 10.0  # [N,2,3] in mm

            targets = format_targets_from_batch(batch)
            have_truth = all(
                key in targets
                for key in (
                    "tar_slice_start_x", "tar_slice_start_y", "tar_slice_start_z",
                    "tar_slice_stop_x", "tar_slice_stop_y", "tar_slice_stop_z",
                )
            )
            truth_xyz = None
            if have_truth:
                truth_start = torch.stack(
                    [
                        targets["tar_slice_start_x"],
                        targets["tar_slice_start_y"],
                        targets["tar_slice_start_z"],
                    ],
                    dim=1,
                ).to(device=pred_xyz.device, dtype=torch.float32)
                truth_stop = torch.stack(
                    [
                        targets["tar_slice_stop_x"],
                        targets["tar_slice_stop_y"],
                        targets["tar_slice_stop_z"],
                    ],
                    dim=1,
                ).to(device=pred_xyz.device, dtype=torch.float32)
                # Targets are stored in normalized ATAR units (x/10,y/10,z/10), same as model output.
                # Convert both truth and predictions to mm for plotting/parity.
                truth_xyz = torch.stack([truth_start, truth_stop], dim=1) * 10.0

            n_rows = min(int(pred_xyz.size(0)), int(target.size(0)))
            if truth_xyz is not None:
                n_rows = min(n_rows, int(truth_xyz.size(0)))
            n_cls = min(int(target.size(1)), len(class_names))
            if n_rows <= 0 or n_cls <= 0:
                continue

            pred_xyz = pred_xyz[:n_rows]
            target = target[:n_rows, :n_cls]
            if truth_xyz is not None:
                truth_xyz = truth_xyz[:n_rows]

            for cls_idx in range(n_cls):
                cls_name = str(class_names[cls_idx])
                mask = target[:, cls_idx] > 0.5
                if int(mask.sum().item()) == 0:
                    continue
                pred_vals = pred_xyz[mask]
                truth_vals = truth_xyz[mask] if truth_xyz is not None else None
                for coord_idx, coord in enumerate(coord_names):
                    coord_key = str(coord)
                    pred_arr = pred_vals[:, :, coord_idx].reshape(-1).detach().cpu().numpy()
                    if pred_arr.size > 0:
                        pred_collected[cls_name][coord_key].append(pred_arr)
                    if truth_vals is not None:
                        truth_arr = truth_vals[:, :, coord_idx].reshape(-1).detach().cpu().numpy()
                        if truth_arr.size > 0:
                            truth_collected[cls_name][coord_key].append(truth_arr)

    pred_map = {
        str(cls): {
            str(coord): (np.concatenate(chunks) if chunks else np.array([], dtype=np.float32))
            for coord, chunks in pred_collected[str(cls)].items()
        }
        for cls in class_names
    }
    truth_map = {
        str(cls): {
            str(coord): (np.concatenate(chunks) if chunks else np.array([], dtype=np.float32))
            for coord, chunks in truth_collected[str(cls)].items()
        }
        for cls in class_names
    }
    pred_counts = {
        str(cls): {str(coord): int(pred_map[str(cls)][str(coord)].size) for coord in coord_names}
        for cls in class_names
    }
    truth_counts = {
        str(cls): {str(coord): int(truth_map[str(cls)][str(coord)].size) for coord in coord_names}
        for cls in class_names
    }
    return pred_map, truth_map, pred_counts, truth_counts


def _build_error_maps(
    *,
    pred_map: Mapping[str, Mapping[str, Sequence[float]]],
    truth_map: Mapping[str, Mapping[str, Sequence[float]]],
    class_names: Sequence[str],
    coord_names: Sequence[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    coord_abs_error: dict[str, dict[str, np.ndarray]] = {}
    euclidean_error: dict[str, np.ndarray] = {}
    for cls in class_names:
        cls_key = str(cls)
        coord_abs_error[cls_key] = {}
        aligned_by_coord: dict[str, np.ndarray] = {}
        for coord in coord_names:
            c_key = str(coord)
            pred_vals = np.asarray(_to_float_list(dict(pred_map.get(cls_key) or {}).get(c_key)), dtype=np.float32)
            truth_vals = np.asarray(_to_float_list(dict(truth_map.get(cls_key) or {}).get(c_key)), dtype=np.float32)
            n = min(int(pred_vals.size), int(truth_vals.size))
            if n <= 0:
                err = np.array([], dtype=np.float32)
            else:
                err = np.abs(pred_vals[:n] - truth_vals[:n]).astype(np.float32, copy=False)
            coord_abs_error[cls_key][c_key] = err
            aligned_by_coord[c_key] = err
        dx = aligned_by_coord.get("x", np.array([], dtype=np.float32))
        dy = aligned_by_coord.get("y", np.array([], dtype=np.float32))
        dz = aligned_by_coord.get("z", np.array([], dtype=np.float32))
        n_xyz = min(int(dx.size), int(dy.size), int(dz.size))
        if n_xyz <= 0:
            euclidean_error[cls_key] = np.array([], dtype=np.float32)
        else:
            euclidean_error[cls_key] = np.sqrt(dx[:n_xyz] ** 2 + dy[:n_xyz] ** 2 + dz[:n_xyz] ** 2).astype(
                np.float32, copy=False
            )
    return coord_abs_error, euclidean_error


@PLOT_REGISTRY_DEF.register("purity_endpoint_pred_truth_histograms_by_particle")
class PurityEndpointPredTruthHistogramsByParticlePlot(BasePlot):
    """Render endpoint coordinate histograms with predicted and truth overlays."""

    name = "purity_endpoint_pred_truth_histograms_by_particle"

    def render(
        self,
        *,
        predicted_by_particle: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        truth_by_particle: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        particle_order: Sequence[str] | None = None,
        coord_order: Sequence[str] = ("x", "y", "z"),
        max_batches: int | None = 100,
        task_weights: Mapping[str, float] | None = None,
        bins: int = 60,
        title_prefix: str = "PURITY Endpoint Coordinate Histograms",
        save_path: str | None = None,
        show: bool = False,
        verbose: bool = False,
    ) -> str | None:
        coords = [str(c) for c in coord_order]
        if not coords:
            return None
        if isinstance(particle_order, Sequence) and not isinstance(particle_order, (str, bytes, bytearray)):
            particles = [str(p) for p in particle_order]
        else:
            particles = []
            for key in list(dict(predicted_by_particle or {}).keys()) + list(dict(truth_by_particle or {}).keys()):
                k = str(key)
                if k not in particles:
                    particles.append(k)
            if not particles:
                particles = ["pion", "muon", "mip"]

        pred_map = dict(predicted_by_particle or {})
        truth_map = dict(truth_by_particle or {})
        if (not pred_map and not truth_map) and module is not None and val_dataloader is not None:
            collected_pred, collected_truth, pred_counts, truth_counts = _collect_endpoint_pred_truth_maps(
                module=module,
                val_dataloader=val_dataloader,
                class_names=particles,
                coord_names=coords,
                max_batches=max_batches,
                task_weights=task_weights,
            )
            pred_map = collected_pred
            truth_map = collected_truth
            if verbose:
                print("endpoint histogram predicted counts:", pred_counts)
                print("endpoint histogram truth counts:", truth_counts)

        if not pred_map and not truth_map:
            return None

        if not particles:
            return None

        save_base: Path | None = Path(str(save_path)) if save_path is not None else None
        first_written: str | None = None
        for particle in particles:
            pred_coords = dict(pred_map.get(particle) or {})
            truth_coords = dict(truth_map.get(particle) or {})
            has_any = False
            for coord in coords:
                pred_vals = [v for v in _to_float_list(pred_coords.get(coord)) if math.isfinite(float(v))]
                truth_vals = [v for v in _to_float_list(truth_coords.get(coord)) if math.isfinite(float(v))]
                if pred_vals or truth_vals:
                    has_any = True
                    break
            if not has_any:
                continue

            total_pred = 0
            total_truth = 0
            for coord in coords:
                total_pred += len([v for v in _to_float_list(pred_coords.get(coord)) if math.isfinite(float(v))])
                total_truth += len([v for v in _to_float_list(truth_coords.get(coord)) if math.isfinite(float(v))])

            fig, axes = plt.subplots(1, len(coords), figsize=(5.2 * len(coords), 4.1), squeeze=False)
            fig.suptitle(
                f"{title_prefix}: {particle} (N_pred={int(total_pred)}, N_truth={int(total_truth)})",
                fontsize=12,
            )
            for idx, coord in enumerate(coords):
                ax = axes[0][idx]
                pred_vals = [v for v in _to_float_list(pred_coords.get(coord)) if math.isfinite(float(v))]
                truth_vals = [v for v in _to_float_list(truth_coords.get(coord)) if math.isfinite(float(v))]
                all_vals = np.asarray(pred_vals + truth_vals, dtype=np.float32)
                bin_edges = None
                if int(all_vals.size) > 0:
                    lo = float(np.min(all_vals))
                    hi = float(np.max(all_vals))
                    if math.isfinite(lo) and math.isfinite(hi):
                        if hi <= lo:
                            delta = 1.0 if abs(lo) < 1e-6 else 0.05 * abs(lo)
                            lo -= delta
                            hi += delta
                        bin_edges = np.linspace(lo, hi, max(2, int(bins) + 1))
                if pred_vals:
                    ax.hist(
                        pred_vals,
                        bins=bin_edges if bin_edges is not None else int(bins),
                        alpha=0.55,
                        color="tab:blue",
                        label=f"pred_{coord}",
                    )
                if truth_vals:
                    ax.hist(
                        truth_vals,
                        bins=bin_edges if bin_edges is not None else int(bins),
                        alpha=0.45,
                        color="tab:orange",
                        label=f"truth_{coord}",
                    )
                if not pred_vals and not truth_vals:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(
                    f"{coord.upper()} (mm) N={int(len(pred_vals) + len(truth_vals))} "
                    f"(pred={int(len(pred_vals))}, truth={int(len(truth_vals))})"
                )
                ax.set_xlabel("Coordinate")
                ax.set_ylabel("Count")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.2)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc="upper right")
            fig.tight_layout()

            per_particle_save: str | None = None
            if save_base is not None:
                safe_particle = "".join(ch for ch in particle if ch.isalnum() or ch in ("_", "-")).strip() or "particle"
                if save_base.suffix:
                    per_particle = save_base.with_name(f"{save_base.stem}_{safe_particle}{save_base.suffix}")
                else:
                    per_particle = save_base / f"endpoint_hist_{safe_particle}.png"
                per_particle.parent.mkdir(parents=True, exist_ok=True)
                per_particle_save = str(per_particle)
                if first_written is None:
                    first_written = per_particle_save

            _save_and_show(fig=fig, save_path=per_particle_save, show=show)

        return first_written if first_written is not None else save_path


@PLOT_REGISTRY_DEF.register("purity_endpoint_pred_histograms_by_particle")
class PurityEndpointPredHistogramsByParticlePlot(BasePlot):
    """Render endpoint coordinate histograms for predictions only."""

    name = "purity_endpoint_pred_histograms_by_particle"

    def render(
        self,
        *,
        predicted_by_particle: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        particle_order: Sequence[str] | None = None,
        coord_order: Sequence[str] = ("x", "y", "z"),
        max_batches: int | None = 100,
        task_weights: Mapping[str, float] | None = None,
        bins: int = 60,
        title_prefix: str = "PURITY Predicted Endpoint Coordinate Histograms",
        save_path: str | None = None,
        show: bool = False,
        verbose: bool = False,
    ) -> str | None:
        pred_map = dict(predicted_by_particle or {})
        if not pred_map and module is not None and val_dataloader is not None:
            coords = [str(c) for c in coord_order]
            if isinstance(particle_order, Sequence) and not isinstance(particle_order, (str, bytes, bytearray)):
                particles = [str(p) for p in particle_order]
            else:
                particles = ["pion", "muon", "mip"]
            collected_pred, _, pred_counts, _ = _collect_endpoint_pred_truth_maps(
                module=module,
                val_dataloader=val_dataloader,
                class_names=particles,
                coord_names=coords,
                max_batches=max_batches,
                task_weights=task_weights,
            )
            pred_map = collected_pred
            if verbose:
                print("endpoint histogram predicted counts:", pred_counts)
        return PurityEndpointPredTruthHistogramsByParticlePlot().render(
            predicted_by_particle=pred_map,
            truth_by_particle={},
            module=None,
            val_dataloader=None,
            particle_order=particle_order,
            coord_order=coord_order,
            max_batches=max_batches,
            task_weights=task_weights,
            bins=bins,
            title_prefix=title_prefix,
            save_path=save_path,
            show=show,
            verbose=verbose,
        )


@PLOT_REGISTRY_DEF.register("purity_endpoint_error_histograms_by_particle")
class PurityEndpointErrorHistogramsByParticlePlot(BasePlot):
    """Render endpoint absolute-error histograms by particle (euclidean + x/y/z)."""

    name = "purity_endpoint_error_histograms_by_particle"

    def render(
        self,
        *,
        predicted_by_particle: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        truth_by_particle: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        particle_order: Sequence[str] | None = None,
        coord_order: Sequence[str] = ("x", "y", "z"),
        max_batches: int | None = 100,
        task_weights: Mapping[str, float] | None = None,
        bins: int = 60,
        title_prefix: str = "PURITY Endpoint Error Histograms",
        save_path: str | None = None,
        show: bool = False,
        verbose: bool = False,
    ) -> str | None:
        coords = [str(c) for c in coord_order]
        if not coords:
            return None
        if isinstance(particle_order, Sequence) and not isinstance(particle_order, (str, bytes, bytearray)):
            particles = [str(p) for p in particle_order]
        else:
            particles = ["pion", "muon", "mip"]

        pred_map = dict(predicted_by_particle or {})
        truth_map = dict(truth_by_particle or {})
        if (not pred_map or not truth_map) and module is not None and val_dataloader is not None:
            collected_pred, collected_truth, pred_counts, truth_counts = _collect_endpoint_pred_truth_maps(
                module=module,
                val_dataloader=val_dataloader,
                class_names=particles,
                coord_names=coords,
                max_batches=max_batches,
                task_weights=task_weights,
            )
            if not pred_map:
                pred_map = collected_pred
            if not truth_map:
                truth_map = collected_truth
            if verbose:
                print("endpoint histogram predicted counts:", pred_counts)
                print("endpoint histogram truth counts:", truth_counts)

        if not pred_map or not truth_map:
            return None

        coord_abs_error, euclidean_error = _build_error_maps(
            pred_map=pred_map,
            truth_map=truth_map,
            class_names=particles,
            coord_names=coords,
        )

        save_base: Path | None = Path(str(save_path)) if save_path is not None else None
        first_written: str | None = None
        for particle in particles:
            cls = str(particle)
            eu = np.asarray(euclidean_error.get(cls, np.array([], dtype=np.float32)), dtype=np.float32)
            x_err = np.asarray(dict(coord_abs_error.get(cls) or {}).get("x", np.array([], dtype=np.float32)), dtype=np.float32)
            y_err = np.asarray(dict(coord_abs_error.get(cls) or {}).get("y", np.array([], dtype=np.float32)), dtype=np.float32)
            z_err = np.asarray(dict(coord_abs_error.get(cls) or {}).get("z", np.array([], dtype=np.float32)), dtype=np.float32)
            if int(eu.size) + int(x_err.size) + int(y_err.size) + int(z_err.size) <= 0:
                continue

            total_n = int(eu.size) + int(x_err.size) + int(y_err.size) + int(z_err.size)
            fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), squeeze=False)
            fig.suptitle(f"{title_prefix}: {cls} (N_total={total_n})", fontsize=12)
            series = [
                ("euclidean", eu, "tab:purple"),
                ("x_abs_error", x_err, "tab:blue"),
                ("y_abs_error", y_err, "tab:orange"),
                ("z_abs_error", z_err, "tab:green"),
            ]
            for ax, (label, vals, color) in zip(axes.reshape(-1), series):
                if int(vals.size) > 0:
                    ax.hist(vals, bins=int(bins), alpha=0.75, color=color)
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{label} (N={int(vals.size)})")
                ax.set_xlabel("Error (mm)")
                ax.set_ylabel("Count")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.2)
            fig.tight_layout()

            per_particle_save: str | None = None
            if save_base is not None:
                safe_particle = "".join(ch for ch in cls if ch.isalnum() or ch in ("_", "-")).strip() or "particle"
                if save_base.suffix:
                    per_particle = save_base.with_name(f"{save_base.stem}_{safe_particle}{save_base.suffix}")
                else:
                    per_particle = save_base / f"endpoint_error_hist_{safe_particle}.png"
                per_particle.parent.mkdir(parents=True, exist_ok=True)
                per_particle_save = str(per_particle)
                if first_written is None:
                    first_written = per_particle_save

            _save_and_show(fig=fig, save_path=per_particle_save, show=show)

        return first_written if first_written is not None else save_path

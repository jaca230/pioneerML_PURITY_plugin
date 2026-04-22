from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch

from pioneerml.evaluation.evaluators.base_evaluator import BaseEvaluator
from pioneerml.evaluation.evaluators.factory.registry import REGISTRY as EVALUATOR_REGISTRY


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        tensor_value = value.detach()
        if tensor_value.numel() == 0:
            return 0.0
        if tensor_value.numel() > 1:
            tensor_value = tensor_value.to(torch.float32).mean()
        return float(tensor_value.cpu().item())
    try:
        return float(value)
    except Exception:
        return 0.0


@EVALUATOR_REGISTRY.register("purity_multilevel")
@EVALUATOR_REGISTRY.register("purity_multitask")
class PurityMultiLevelEvaluator(BaseEvaluator):
    """Multi-head evaluator that reports aggregated multi-task loss terms."""

    default_metric_names: tuple[str, ...] = ()
    default_plot_names: tuple[str, ...] = ("loss_curves", "purity_pdg_accuracy_curves")

    def build_context(
        self,
        *,
        module,
        loader,
        config: Mapping[str, object],
    ) -> dict[str, object]:
        module.eval()

        device = next(module.parameters()).device
        total_loss = 0.0
        total_samples = 0

        term_sums: dict[str, float] = defaultdict(float)
        term_weights: dict[str, int] = defaultdict(int)

        threshold = float(config.get("threshold", 0.5))
        event_label_total = 0
        event_label_equal = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                raw_preds = module(batch)
                loss, terms = module.compute_loss(raw_preds, batch)

                batch_size = int(module._get_batch_size(batch))
                total_loss += float(loss.detach().cpu().item()) * batch_size
                total_samples += batch_size

                for key, value in dict(terms).items():
                    if str(key) == "loss":
                        continue
                    metric_name = str(key)
                    term_sums[metric_name] += _to_float(value) * batch_size
                    term_weights[metric_name] += batch_size

                # Optional quick event-binary sanity metric on primary output.
                if bool(config.get("event_binary_metrics", True)):
                    try:
                        preds = module.primary_predictions(raw_preds)
                        target = module.primary_target(batch, preds)
                        probs = torch.sigmoid(preds.detach())
                        pred_bin = (probs >= threshold).to(torch.int64)
                        target_bin = target.detach().to(torch.int64)
                        if pred_bin.ndim == 2 and pred_bin.shape[1] == 1:
                            pred_bin = pred_bin.view(-1)
                        if target_bin.ndim == 2 and target_bin.shape[1] == 1:
                            target_bin = target_bin.view(-1)
                        if pred_bin.shape == target_bin.shape:
                            event_label_total += int(pred_bin.numel())
                            event_label_equal += int((pred_bin == target_bin).sum().item())
                    except Exception:
                        # Keep evaluator robust if a given run omits event-level targets.
                        pass

        if total_samples <= 0:
            raise RuntimeError("No samples available for evaluation.")

        base_metrics: dict[str, object] = {
            "loss": float(total_loss / total_samples),
            "threshold": float(threshold),
            "train_loss_history": list(module.train_epoch_loss_history),
            "train_loss_history_total_points": len(list(module.train_epoch_loss_history)),
            "val_loss_history": list(module.val_epoch_loss_history),
            "val_loss_history_total_points": len(list(module.val_epoch_loss_history)),
        }

        for key, weighted_sum in term_sums.items():
            denom = max(1, int(term_weights.get(key, 0)))
            base_metrics[key] = float(weighted_sum / denom)

        if event_label_total > 0:
            base_metrics["event_binary_accuracy"] = float(event_label_equal) / float(event_label_total)
            base_metrics["event_binary_total"] = int(event_label_total)

        plot_path = self.resolve_plot_path(dict(config))
        plot_dir = config.get("plot_dir")
        staged_plot_path = None
        phase1_plot_path = None
        phase2_plot_path = None
        phase3_plot_path = None
        pdg_plot_path = None
        if plot_path is not None:
            plot_file = Path(str(plot_path))
            parent = plot_file.parent
            stem = plot_file.stem
            suffix = plot_file.suffix or ".png"
            staged_plot_path = str(parent / f"{stem}_staged{suffix}")
            phase1_plot_path = str(parent / f"{stem}_phase1{suffix}")
            phase2_plot_path = str(parent / f"{stem}_phase2{suffix}")
            phase3_plot_path = str(parent / f"{stem}_phase3{suffix}")
            pdg_plot_path = str(parent / f"{stem}_pdg_accuracy{suffix}")
        elif isinstance(plot_dir, str) and plot_dir.strip():
            parent = Path(plot_dir)
            staged_plot_path = str(parent / "purity_staged_loss_curves.png")
            phase1_plot_path = str(parent / "purity_phase_1_loss_curves.png")
            phase2_plot_path = str(parent / "purity_phase_2_loss_curves.png")
            phase3_plot_path = str(parent / "purity_phase_3_loss_curves.png")
            pdg_plot_path = str(parent / "purity_pdg_accuracy_curves.png")

        return {
            "metric_context": {},
            "plot_kwargs_by_name": {
                "loss_curves": {
                    "train_losses": module,
                    "save_path": plot_path,
                    "show": False,
                },
                "purity_staged_loss_curves": {
                    "module": module,
                    "save_path": staged_plot_path,
                    "show": False,
                },
                "purity_phase_1_loss_curves": {
                    "module": module,
                    "save_path": phase1_plot_path,
                    "show": False,
                },
                "purity_phase_2_loss_curves": {
                    "module": module,
                    "save_path": phase2_plot_path,
                    "show": False,
                },
                "purity_phase_3_loss_curves": {
                    "module": module,
                    "save_path": phase3_plot_path,
                    "show": False,
                },
                "purity_pdg_accuracy_curves": {
                    "module": module,
                    "split": "train",
                    "save_path": pdg_plot_path,
                    "show": False,
                },
            },
            "base_metrics": base_metrics,
        }

    def finalize_results(
        self,
        *,
        results: dict[str, object],
        context: Mapping[str, object],
        config: Mapping[str, object],
    ) -> dict[str, object]:
        _ = context
        _ = config
        loss_plot_path = results.get("loss_curves_path")
        if isinstance(loss_plot_path, str):
            results["loss_plot_path"] = loss_plot_path
        pdg_plot_path = results.get("purity_pdg_accuracy_curves_path")
        if isinstance(pdg_plot_path, str):
            results["pdg_accuracy_plot_path"] = pdg_plot_path
        return results

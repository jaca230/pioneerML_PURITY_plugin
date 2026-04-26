from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from pioneerml.evaluation.plots.base_plot import BasePlot

from .common import _save_and_show

CLASS_NAMES = ("pion", "muon", "mip")
TIME_SENTINEL_DEFAULT = -499.0


def _cat(xs: list[np.ndarray]) -> np.ndarray:
    if not xs:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(xs).astype(np.float32, copy=False)


def _as_prob(t: torch.Tensor) -> torch.Tensor:
    t = t.float().view(-1) if t.dim() == 1 else t.float()
    if int(t.numel()) == 0:
        return t
    t_min = float(t.detach().min().item())
    t_max = float(t.detach().max().item())
    if t_min >= 0.0 and t_max <= 1.0:
        return t
    return torch.sigmoid(t)


def _align_node_like_target(target: torch.Tensor, batch: Any) -> torch.Tensor:
    x = batch.x
    is_atar = (x[:, 5] > 0.5) | (x[:, 6] > 0.5)
    tar = target.float()
    if tar.dim() == 2 and int(tar.size(0)) == int(x.size(0)):
        tar = tar[is_atar]
    return tar


def _truth_positron_time_proxy(batch: Any) -> torch.Tensor:
    x = batch.x
    b = batch.batch
    is_atar = (x[:, 5] > 0.5) | (x[:, 6] > 0.5)
    n_graphs = int(b.max().item()) + 1 if int(b.numel()) > 0 else 0
    if n_graphs == 0 or int(is_atar.sum().item()) == 0:
        return torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=x.device)

    trig = getattr(batch, "is_trigger_target", None)
    node = getattr(batch, "atar_node_pdg_target", None)
    if not isinstance(trig, torch.Tensor) or not isinstance(node, torch.Tensor):
        return torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=x.device)
    if int(node.dim()) != 2 or int(node.size(1)) < 3:
        return torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=x.device)

    trig = trig.view(-1).float()
    trig_atar = trig[is_atar] if int(trig.numel()) == int(x.size(0)) else trig
    node_atar = node[is_atar] if int(node.size(0)) == int(x.size(0)) else node

    n = min(int(node_atar.size(0)), int(trig_atar.numel()), int(is_atar.sum().item()))
    if n <= 0:
        return torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=x.device)

    atar_times = x[is_atar, 4][:n]
    atar_batch = b[is_atar][:n]
    mask = (trig_atar[:n] > 0.5) & (node_atar[:n, 2] > 0.5)

    num = torch.zeros((n_graphs,), dtype=torch.float32, device=x.device)
    den = torch.zeros((n_graphs,), dtype=torch.float32, device=x.device)
    num.index_add_(0, atar_batch, atar_times * mask.float())
    den.index_add_(0, atar_batch, mask.float())

    out = torch.full((n_graphs,), float("nan"), dtype=torch.float32, device=x.device)
    good = den > 0.5
    out[good] = num[good] / den[good]
    return out


def _truth_angle_vec_per_graph(batch: Any, *, num_graphs: int, device: torch.device) -> torch.Tensor:
    out = torch.full((max(0, int(num_graphs)), 3), float("nan"), dtype=torch.float32, device=device)
    angle = getattr(batch, "atar_angle_target", None)
    ptr = getattr(batch, "atar_slice_ptr", None)
    if not isinstance(angle, torch.Tensor) or not isinstance(ptr, torch.Tensor):
        return out
    if int(angle.dim()) != 2 or int(angle.size(1)) < 3 or int(ptr.numel()) < 2:
        return out

    g_max = min(int(num_graphs), int(ptr.numel()) - 1)
    for g in range(g_max):
        s0 = int(ptr[g].item())
        s1 = int(ptr[g + 1].item())
        if s1 <= s0 or s0 < 0 or s0 >= int(angle.size(0)):
            continue
        v = angle[s0, :3].to(dtype=torch.float32, device=device)
        if not bool(torch.isfinite(v).all().item()):
            continue
        nrm = float(v.norm().item())
        if nrm <= 1e-8:
            continue
        out[g] = v / nrm
    return out


def _resolve_energy_scale_for_plot(*, pred: np.ndarray, truth: np.ndarray, module: Any) -> tuple[np.ndarray, float, str]:
    pred_plot = np.asarray(pred, dtype=np.float32).copy()
    truth_ref = np.asarray(truth, dtype=np.float32)
    if int(pred_plot.size) <= 0 or int(truth_ref.size) <= 0:
        return pred_plot, 1.0, "empty"

    finite = np.isfinite(pred_plot) & np.isfinite(truth_ref)
    if not np.any(finite):
        return pred_plot, 1.0, "non_finite"

    p = np.abs(pred_plot[finite])
    t = np.abs(truth_ref[finite])
    p90 = float(np.percentile(p, 90.0)) if int(p.size) > 0 else 0.0
    t90 = float(np.percentile(t, 90.0)) if int(t.size) > 0 else 0.0
    if p90 <= 0.0 or t90 <= 0.0:
        return pred_plot, 1.0, "degenerate"

    amp_ratio = t90 / max(p90, 1e-6)
    if amp_ratio < 8.0:
        return pred_plot, 1.0, "already_aligned"

    ratio = float("nan")
    pos = (p > 1e-8) & (t > 1e-8)
    if np.any(pos):
        ratio = float(np.median(t[pos]) / max(np.median(p[pos]), 1e-8))

    candidate = 70.0
    impl = getattr(getattr(module, "model", None), "impl", None)
    norm = getattr(impl, "norm_e_lyso", None) if impl is not None else None
    if norm is not None:
        try:
            norm_f = float(norm)
            if np.isfinite(norm_f) and norm_f > 1.0:
                candidate = norm_f
        except Exception:
            pass

    if np.isfinite(ratio) and ratio > 8.0:
        if 0.5 * candidate <= ratio <= 2.0 * candidate:
            scale = candidate
            reason = "lyso_norm"
        else:
            scale = ratio
            reason = "median_ratio"
    else:
        scale = candidate
        reason = "lyso_norm_fallback"

    pred_plot *= float(scale)
    return pred_plot.astype(np.float32, copy=False), float(scale), reason


def _event_token_truth_from_batch(*, out: Mapping[str, Any], batch: Any) -> tuple[torch.Tensor | None, dict[str, int]]:
    logits = out.get("unified_event_logits")
    if not isinstance(logits, torch.Tensor):
        return None, {
            "event_tokens_total": 0,
            "event_tokens_atar": 0,
            "event_tokens_lyso": 0,
            "event_tokens_truth_defined": 0,
            "event_tokens_truth_graph_fallback": 0,
        }

    logits_1d = logits.view(-1)
    n_tokens = int(logits_1d.numel())
    device = logits_1d.device
    truth = torch.full((n_tokens,), float("nan"), dtype=torch.float32, device=device)
    token_graph = torch.full((n_tokens,), -1, dtype=torch.long, device=device)

    meta = {
        "event_tokens_total": int(n_tokens),
        "event_tokens_atar": 0,
        "event_tokens_lyso": 0,
        "event_tokens_truth_defined": 0,
        "event_tokens_truth_graph_fallback": 0,
    }

    x = getattr(batch, "x", None)
    b = getattr(batch, "batch", None)
    if not isinstance(x, torch.Tensor) or not isinstance(b, torch.Tensor) or int(b.numel()) <= 0:
        meta["event_tokens_truth_defined"] = 0
        return truth, meta

    x = x.to(device=device)
    b = b.to(device=device, dtype=torch.long)
    num_graphs = int(b.max().item()) + 1
    trigger = getattr(batch, "is_trigger_target", None)
    trigger_t = trigger.to(device=device, dtype=torch.float32).view(-1) if isinstance(trigger, torch.Tensor) else None

    is_atar = (x[:, 5] > 0.5) | (x[:, 6] > 0.5)
    is_lyso = x[:, 7] > 0.5

    n_atar = out.get("unified_num_atar_tokens", 0)
    if isinstance(n_atar, torch.Tensor):
        n_atar = int(n_atar.view(-1)[0].item()) if int(n_atar.numel()) > 0 else 0
    else:
        n_atar = int(n_atar)
    n_atar = max(0, min(n_atar, n_tokens))

    if n_atar > 0:
        valid_slice_mask = out.get("valid_slice_mask")
        num_slices_max = out.get("num_slices_max", 0)
        if isinstance(num_slices_max, torch.Tensor):
            num_slices_max = int(num_slices_max.view(-1)[0].item()) if int(num_slices_max.numel()) > 0 else 0
        else:
            num_slices_max = int(num_slices_max)

        if isinstance(valid_slice_mask, torch.Tensor):
            valid_slice_mask = valid_slice_mask.to(device=device, dtype=torch.bool).view(-1)
            if num_slices_max <= 0 and num_graphs > 0 and int(valid_slice_mask.numel()) > 0:
                num_slices_max = max(1, int(valid_slice_mask.numel()) // max(1, num_graphs))

            valid_slice_idx = torch.nonzero(valid_slice_mask, as_tuple=False).view(-1)
            atar_token_graph = valid_slice_idx // max(1, num_slices_max)
            atar_slots = int(atar_token_graph.numel())
            m_tok = min(int(n_atar), atar_slots)
            if m_tok > 0:
                token_graph[:m_tok] = atar_token_graph[:m_tok]
                meta["event_tokens_atar"] = int(m_tok)

            if trigger_t is not None and bool(is_atar.any().item()) and atar_slots > 0 and num_slices_max > 0:
                atar_batch = b[is_atar]
                atar_slice = x[is_atar, 8].to(dtype=torch.long)
                global_slice_ids = atar_batch * int(num_slices_max) + atar_slice
                if int(valid_slice_mask.numel()) > 0:
                    global_slice_ids = global_slice_ids.clamp(min=0, max=int(valid_slice_mask.numel()) - 1)
                    hit_valid = valid_slice_mask[global_slice_ids]
                    if bool(hit_valid.any().item()):
                        mapped_slice_indices = torch.cumsum(valid_slice_mask.long(), dim=0) - 1
                        idx = mapped_slice_indices[global_slice_ids[hit_valid]]
                        tar_atar = trigger_t[is_atar]
                        y = tar_atar[hit_valid]
                        valid_idx = (idx >= 0) & (idx < atar_slots)
                        if bool(valid_idx.any().item()):
                            idx = idx[valid_idx]
                            y = y[valid_idx]
                            num = torch.zeros((atar_slots,), dtype=torch.float32, device=device)
                            den = torch.zeros((atar_slots,), dtype=torch.float32, device=device)
                            num.index_add_(0, idx, y)
                            den.index_add_(0, idx, torch.ones_like(y))
                            atar_truth = torch.full((atar_slots,), float("nan"), dtype=torch.float32, device=device)
                            good = den > 0.5
                            atar_truth[good] = num[good] / den[good]
                            m_write = min(atar_slots, n_atar)
                            if m_write > 0:
                                truth[:m_write] = atar_truth[:m_write]

    n_remaining = max(0, n_tokens - n_atar)
    if n_remaining > 0 and bool(is_lyso.any().item()):
        lyso_batch = b[is_lyso]
        n_lyso_hits = int(lyso_batch.numel())
        if n_lyso_hits > 0:
            graph_has_lyso = torch.zeros((num_graphs,), dtype=torch.bool, device=device)
            graph_has_lyso[lyso_batch] = True
            lyso_graph_ids = torch.nonzero(graph_has_lyso, as_tuple=False).view(-1)
            n_lyso_graphs = int(lyso_graph_ids.numel())

            assign = out.get("lyso_soft_assignments")
            k = int(assign.size(1)) if isinstance(assign, torch.Tensor) and int(assign.dim()) >= 2 else 0
            if n_lyso_graphs > 0 and k > 0:
                lyso_token_graph = lyso_graph_ids.repeat_interleave(k)
                n_expected = int(lyso_token_graph.numel())
                n_take = min(n_remaining, n_expected)
                if n_take > 0:
                    token_graph[n_atar : n_atar + n_take] = lyso_token_graph[:n_take]
                    meta["event_tokens_lyso"] = int(n_take)

                if trigger_t is not None and n_take > 0:
                    lyso_targets = trigger_t[is_lyso]
                    n_hits = min(int(lyso_targets.numel()), int(assign.size(0)), int(lyso_batch.numel()))
                    if n_hits > 0:
                        lyso_targets = lyso_targets[:n_hits]
                        lyso_batch = lyso_batch[:n_hits]
                        assign = assign[:n_hits, :k].to(device=device, dtype=torch.float32)

                        mapped_graph_indices = torch.cumsum(graph_has_lyso.long(), dim=0) - 1
                        lyso_mapped_batch = mapped_graph_indices[lyso_batch]

                        effective = assign
                        seed_beta = out.get("lyso_seed_beta")
                        if isinstance(seed_beta, torch.Tensor) and int(seed_beta.numel()) >= n_lyso_graphs * k:
                            beta = seed_beta.to(device=device, dtype=torch.float32).view(-1)[: n_lyso_graphs * k].view(
                                n_lyso_graphs, k
                            )
                            effective = effective * beta[lyso_mapped_batch]

                        numer = torch.zeros((n_lyso_graphs, k), dtype=torch.float32, device=device)
                        denom = torch.zeros((n_lyso_graphs, k), dtype=torch.float32, device=device)
                        cluster_idx = torch.arange(k, device=device, dtype=torch.long).unsqueeze(0)
                        flat_idx = lyso_mapped_batch.unsqueeze(1) * k + cluster_idx
                        numer_flat = numer.view(-1)
                        denom_flat = denom.view(-1)
                        numer_flat.index_add_(
                            0,
                            flat_idx.reshape(-1),
                            (effective * lyso_targets.unsqueeze(1)).reshape(-1),
                        )
                        denom_flat.index_add_(0, flat_idx.reshape(-1), effective.reshape(-1))

                        lyso_truth = torch.full((n_lyso_graphs, k), float("nan"), dtype=torch.float32, device=device)
                        good = denom > 1e-6
                        lyso_truth[good] = numer[good] / denom[good]
                        lyso_truth_1d = lyso_truth.view(-1)
                        truth[n_atar : n_atar + n_take] = lyso_truth_1d[:n_take]

    graph_truth = None
    for field in ("y_event", "y_graph", "y"):
        value = getattr(batch, field, None)
        if isinstance(value, torch.Tensor) and int(value.numel()) > 0:
            graph_truth = value.to(device=device, dtype=torch.float32).view(-1)
            break

    before_defined = int(torch.isfinite(truth).sum().item())
    if isinstance(graph_truth, torch.Tensor) and int(graph_truth.numel()) > 0:
        valid_tok_graph = (token_graph >= 0) & (token_graph < int(graph_truth.numel()))
        fill = torch.isnan(truth) & valid_tok_graph
        if bool(fill.any().item()):
            truth[fill] = graph_truth[token_graph[fill]]

    after_defined = int(torch.isfinite(truth).sum().item())
    meta["event_tokens_truth_defined"] = int(after_defined)
    meta["event_tokens_truth_graph_fallback"] = int(max(0, after_defined - before_defined))
    return truth, meta


def collect_purity_target_diagnostics(
    *,
    module: Any,
    val_dataloader: Any,
    task_weights: Mapping[str, float] | None = None,
    max_batches: int = 200,
    verbose: bool = False,
) -> dict[str, Any]:
    from pioneerml_purity_plugin.purity.losses.utils.unified_training_components import format_targets_from_batch

    if module is None or val_dataloader is None:
        raise ValueError("collect_purity_target_diagnostics requires module and val_dataloader.")

    model = module.model.eval()
    active_task_weights = (
        dict(task_weights) if isinstance(task_weights, Mapping) else getattr(module, "_task_weights", None)
    )

    node_conf = np.zeros((3, 3), dtype=np.int64)
    slice_conf = np.zeros((3, 3), dtype=np.int64)

    trig_scores: list[np.ndarray] = []
    trig_truth: list[np.ndarray] = []
    multi_scores: list[np.ndarray] = []
    multi_truth: list[np.ndarray] = []
    event_scores: list[np.ndarray] = []
    event_truth: list[np.ndarray] = []

    pion_euclid: list[np.ndarray] = []
    pion_dx: list[np.ndarray] = []
    pion_dy: list[np.ndarray] = []
    pion_dz: list[np.ndarray] = []

    theta_diff_deg: list[np.ndarray] = []
    theta_pred_deg: list[np.ndarray] = []
    theta_truth_deg: list[np.ndarray] = []
    angle_cos_err: list[np.ndarray] = []
    energy_pred: list[np.ndarray] = []
    energy_truth: list[np.ndarray] = []
    time_pred: list[np.ndarray] = []
    time_truth_proxy: list[np.ndarray] = []
    accepted_vals: list[np.ndarray] = []

    stats: dict[str, int | float] = {
        "batches_processed": 0,
        "graphs_seen": 0,
        "trig_pairs_total": 0,
        "multi_pairs_total": 0,
        "event_pairs_total": 0,
        "event_tokens_total": 0,
        "event_tokens_atar": 0,
        "event_tokens_lyso": 0,
        "event_tokens_truth_defined": 0,
        "event_tokens_truth_graph_fallback": 0,
        "pion_graph_pairs_total": 0,
        "pion_graph_pairs_valid_truth_mask": 0,
        "pion_graph_pairs_used": 0,
        "theta_graph_candidates": 0,
        "theta_graph_with_truth": 0,
        "energy_pairs_total": 0,
        "time_pairs_total": 0,
        "time_pairs_finite": 0,
        "time_pairs_sentinel_removed": 0,
        "time_pairs_used": 0,
        "time_pairs_sentinel_fraction_of_finite": 0.0,
    }

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            if i >= int(max_batches):
                break
            device = next(model.parameters()).device
            batch = batch.to(device)
            stats["batches_processed"] = int(stats["batches_processed"]) + 1
            n_graphs_in_batch = int(getattr(batch, "num_graphs", 0))
            if n_graphs_in_batch <= 0 and hasattr(batch, "batch") and int(batch.batch.numel()) > 0:
                n_graphs_in_batch = int(batch.batch.max().item()) + 1
            stats["graphs_seen"] = int(stats["graphs_seen"]) + max(0, n_graphs_in_batch)

            if active_task_weights:
                out = model(batch, task_weights=active_task_weights)
            else:
                out = model(batch)
            targets = format_targets_from_batch(batch)

            if (
                "atar_node_pdg" in out
                and hasattr(batch, "atar_node_pdg_target")
                and isinstance(batch.atar_node_pdg_target, torch.Tensor)
            ):
                pred = _as_prob(out["atar_node_pdg"])
                tar = _align_node_like_target(batch.atar_node_pdg_target, batch)
                n = min(int(pred.size(0)), int(tar.size(0)))
                if n > 0 and int(tar.size(1)) >= 3:
                    pred_cls = torch.argmax(pred[:n, :3], dim=1).detach().cpu().numpy()
                    tar_cls = torch.argmax(tar[:n, :3], dim=1).detach().cpu().numpy()
                    for t, p in zip(tar_cls, pred_cls, strict=False):
                        if 0 <= int(t) < 3 and 0 <= int(p) < 3:
                            node_conf[int(t), int(p)] += 1

            if "atar_slice_pdg" in out and "tar_slice_pdg" in targets:
                pred = _as_prob(out["atar_slice_pdg"])
                tar = targets["tar_slice_pdg"].float()
                n = min(int(pred.size(0)), int(tar.size(0)))
                if n > 0 and int(tar.size(1)) >= 3:
                    pred_cls = torch.argmax(pred[:n, :3], dim=1).detach().cpu().numpy()
                    tar_cls = torch.argmax(tar[:n, :3], dim=1).detach().cpu().numpy()
                    for t, p in zip(tar_cls, pred_cls, strict=False):
                        if 0 <= int(t) < 3 and 0 <= int(p) < 3:
                            slice_conf[int(t), int(p)] += 1

            if "atar_trigger_logits" in out and "tar_slice_trigger" in targets:
                s = _as_prob(out["atar_trigger_logits"].view(-1)).detach().cpu().numpy()
                y = targets["tar_slice_trigger"].float().view(-1).detach().cpu().numpy()
                m = min(len(s), len(y))
                if m > 0:
                    trig_scores.append(s[:m])
                    trig_truth.append(y[:m])
                    stats["trig_pairs_total"] = int(stats["trig_pairs_total"]) + int(m)

            if (
                "atar_slice_multi" in out
                and hasattr(batch, "atar_slice_multi_target")
                and isinstance(batch.atar_slice_multi_target, torch.Tensor)
            ):
                s = _as_prob(out["atar_slice_multi"].view(-1)).detach().cpu().numpy()
                y = batch.atar_slice_multi_target.float().view(-1).detach().cpu().numpy()
                m = min(len(s), len(y))
                if m > 0:
                    multi_scores.append(s[:m])
                    multi_truth.append(y[:m])
                    stats["multi_pairs_total"] = int(stats["multi_pairs_total"]) + int(m)

            if "atar_pion_stop" in out and "tar_pion_stop_xyz" in targets:
                pred = out["atar_pion_stop"].float()
                tar = targets["tar_pion_stop_xyz"].float()
                n = min(int(pred.size(0)), int(tar.size(0)))
                if n > 0:
                    pred = pred[:n]
                    tar = tar[:n]
                    stats["pion_graph_pairs_total"] = int(stats["pion_graph_pairs_total"]) + int(n)

                    if "tar_pion_stop_valid_graph" in targets:
                        valid = targets["tar_pion_stop_valid_graph"].view(-1)[:n] > 0.5
                    else:
                        valid = torch.isfinite(tar).all(dim=1)
                    valid = valid & torch.isfinite(pred).all(dim=1) & torch.isfinite(tar).all(dim=1)
                    stats["pion_graph_pairs_valid_truth_mask"] = int(stats["pion_graph_pairs_valid_truth_mask"]) + int(
                        valid.sum().item()
                    )

                    if bool(valid.any().item()):
                        pred = pred[valid]
                        tar = tar[valid]
                        if int(pred.size(0)) > 0:
                            diff_mm = (pred - tar) * 10.0
                            pion_dx.append(diff_mm[:, 0].detach().cpu().numpy())
                            pion_dy.append(diff_mm[:, 1].detach().cpu().numpy())
                            pion_dz.append(diff_mm[:, 2].detach().cpu().numpy())
                            pion_euclid.append(torch.linalg.norm(diff_mm, dim=1).detach().cpu().numpy())
                            stats["pion_graph_pairs_used"] = int(stats["pion_graph_pairs_used"]) + int(pred.size(0))

            has_pos = batch.has_trigger_positron.bool() if hasattr(batch, "has_trigger_positron") else None
            if isinstance(has_pos, torch.Tensor) and "atar_positron_dir" in out:
                pred_dirs = out["atar_positron_dir"].float()
                n = min(int(pred_dirs.size(0)), int(has_pos.numel()))
                if n > 0:
                    pred_dirs = pred_dirs[:n]
                    pos_mask = has_pos[:n]
                    stats["theta_graph_candidates"] = int(stats["theta_graph_candidates"]) + int(pos_mask.sum().item())
                    if bool(pos_mask.any().item()):
                        truth_dirs_graph = _truth_angle_vec_per_graph(batch, num_graphs=n, device=pred_dirs.device)
                        pred_sel = pred_dirs[pos_mask]
                        truth_sel = truth_dirs_graph[pos_mask]
                        pred_finite = torch.isfinite(pred_sel).all(dim=1)
                        truth_finite = torch.isfinite(truth_sel).all(dim=1)
                        truth_norm = truth_sel.norm(dim=1)
                        valid = pred_finite & truth_finite & (truth_norm > 1e-8)
                        if bool(valid.any().item()):
                            pred_unit = pred_sel[valid].div(pred_sel[valid].norm(dim=1, keepdim=True).clamp(min=1e-9))
                            truth_unit = truth_sel[valid].div(truth_sel[valid].norm(dim=1, keepdim=True).clamp(min=1e-9))
                            pred_theta = torch.acos(pred_unit[:, 2].clamp(-1, 1))
                            truth_theta = torch.acos(truth_unit[:, 2].clamp(-1, 1))
                            theta_pred_deg.append(pred_theta.rad2deg().detach().cpu().numpy())
                            theta_truth_deg.append(truth_theta.rad2deg().detach().cpu().numpy())
                            theta_diff_deg.append((truth_theta - pred_theta).rad2deg().detach().cpu().numpy())
                            angle_cos_err.append(
                                (1.0 - F.cosine_similarity(pred_unit, truth_unit, dim=1)).detach().cpu().numpy()
                            )
                            stats["theta_graph_with_truth"] = int(stats["theta_graph_with_truth"]) + int(valid.sum().item())

            if "unified_event_logits" in out:
                score = _as_prob(out["unified_event_logits"].view(-1)).detach().cpu().numpy()
                truth_t, event_meta = _event_token_truth_from_batch(out=out, batch=batch)
                for key in (
                    "event_tokens_total",
                    "event_tokens_atar",
                    "event_tokens_lyso",
                    "event_tokens_truth_defined",
                    "event_tokens_truth_graph_fallback",
                ):
                    stats[key] = int(stats.get(key, 0)) + int(event_meta.get(key, 0))
                if isinstance(truth_t, torch.Tensor):
                    truth_np = truth_t.detach().cpu().numpy()
                    m = min(len(score), len(truth_np))
                    if m > 0:
                        s = score[:m]
                        y = truth_np[:m]
                        valid = np.isfinite(s) & np.isfinite(y)
                        if bool(np.any(valid)):
                            event_scores.append(s[valid])
                            event_truth.append(y[valid])
                            stats["event_pairs_total"] = int(stats["event_pairs_total"]) + int(np.sum(valid))

            if "summary_positron_energy" in out and hasattr(batch, "positron_initial_energy_target"):
                pred_e = out["summary_positron_energy"].detach().cpu().view(-1).numpy()
                tar_e = batch.positron_initial_energy_target.detach().cpu().view(-1).numpy()
                m = min(len(pred_e), len(tar_e))
                if m > 0:
                    energy_pred.append(pred_e[:m])
                    energy_truth.append(tar_e[:m])
                    stats["energy_pairs_total"] = int(stats["energy_pairs_total"]) + int(m)

            if "summary_positron_time" in out:
                pred_t = out["summary_positron_time"].detach().cpu().view(-1)
                tar_t = _truth_positron_time_proxy(batch).detach().cpu().view(-1)
                m = min(int(pred_t.numel()), int(tar_t.numel()))
                if m > 0:
                    time_pred.append(pred_t[:m].numpy())
                    time_truth_proxy.append(tar_t[:m].numpy())
                    stats["time_pairs_total"] = int(stats["time_pairs_total"]) + int(m)

            if "summary_accepted" in out:
                accepted_vals.append(out["summary_accepted"].detach().cpu().view(-1).numpy())

    out_diag = {
        "class_names": CLASS_NAMES,
        "node_conf": node_conf,
        "slice_conf": slice_conf,
        "trig_scores": _cat(trig_scores),
        "trig_truth": _cat(trig_truth),
        "multi_scores": _cat(multi_scores),
        "multi_truth": _cat(multi_truth),
        "event_scores": _cat(event_scores),
        "event_truth": _cat(event_truth),
        "pion_euclid": _cat(pion_euclid),
        "pion_dx": _cat(pion_dx),
        "pion_dy": _cat(pion_dy),
        "pion_dz": _cat(pion_dz),
        "theta_diff_deg": _cat(theta_diff_deg),
        "theta_pred_deg": _cat(theta_pred_deg),
        "theta_truth_deg": _cat(theta_truth_deg),
        "angle_cos_err": _cat(angle_cos_err),
        "energy_pred": _cat(energy_pred),
        "energy_truth": _cat(energy_truth),
        "time_pred": _cat(time_pred),
        "time_truth_proxy": _cat(time_truth_proxy),
        "accepted_vals": _cat(accepted_vals),
        "time_sentinel_threshold": float(TIME_SENTINEL_DEFAULT),
    }

    pred_e = np.asarray(out_diag.get("energy_pred"), dtype=np.float32)
    truth_e = np.asarray(out_diag.get("energy_truth"), dtype=np.float32)
    m_e = min(int(pred_e.size), int(truth_e.size))
    pred_e = pred_e[:m_e]
    truth_e = truth_e[:m_e]
    finite_e = np.isfinite(pred_e) & np.isfinite(truth_e)
    pred_e_valid = pred_e[finite_e]
    truth_e_valid = truth_e[finite_e]
    # Keep raw units for diagnostics plots; avoid heuristic scaling.
    out_diag["energy_pred_plot"] = pred_e_valid.astype(np.float32, copy=False)
    out_diag["energy_truth_plot"] = truth_e_valid.astype(np.float32, copy=False)
    out_diag["energy_scale_factor"] = 1.0
    out_diag["energy_scale_reason"] = "raw_units"
    stats["energy_pairs_finite"] = int(finite_e.sum())

    pred_t = np.asarray(out_diag.get("time_pred"), dtype=np.float32)
    truth_t = np.asarray(out_diag.get("time_truth_proxy"), dtype=np.float32)
    m_t = min(int(pred_t.size), int(truth_t.size))
    pred_t = pred_t[:m_t]
    truth_t = truth_t[:m_t]
    finite_t = np.isfinite(pred_t) & np.isfinite(truth_t)
    sentinel_t = (pred_t <= float(TIME_SENTINEL_DEFAULT)) | (truth_t <= float(TIME_SENTINEL_DEFAULT))
    keep_t = finite_t & (~sentinel_t)
    out_diag["time_pred_filtered"] = pred_t[keep_t].astype(np.float32, copy=False)
    out_diag["time_truth_proxy_filtered"] = truth_t[keep_t].astype(np.float32, copy=False)
    stats["time_pairs_finite"] = int(finite_t.sum())
    stats["time_pairs_sentinel_removed"] = int((finite_t & sentinel_t).sum())
    stats["time_pairs_used"] = int(keep_t.sum())
    stats["time_pairs_sentinel_fraction_of_finite"] = (
        float(stats["time_pairs_sentinel_removed"]) / max(1.0, float(stats["time_pairs_finite"]))
    )

    out_diag["sample_stats"] = stats

    if verbose:
        print(
            "target_diag sizes:",
            {
                k: (int(v.size) if isinstance(v, np.ndarray) else "matrix")
                for k, v in out_diag.items()
                if k not in {"class_names", "sample_stats"}
            },
        )
        print("target_diag stats:", stats)
    return out_diag


def resolve_target_diagnostics(
    *,
    diagnostics: Mapping[str, Any] | None = None,
    module: Any | None = None,
    val_dataloader: Any | None = None,
    task_weights: Mapping[str, float] | None = None,
    max_batches: int = 200,
) -> dict[str, Any] | None:
    if isinstance(diagnostics, Mapping):
        return dict(diagnostics)
    if module is None or val_dataloader is None:
        return None
    return collect_purity_target_diagnostics(
        module=module,
        val_dataloader=val_dataloader,
        task_weights=task_weights,
        max_batches=max_batches,
        verbose=False,
    )


def per_class_accuracy(conf: np.ndarray) -> np.ndarray:
    denom = conf.sum(axis=1).clip(min=1)
    return np.diag(conf) / denom


class BinaryScoreHistogramBasePlot(BasePlot):
    score_key: str = ""
    truth_key: str = ""
    title_default: str = "Binary Score Distribution"

    def render(
        self,
        *,
        diagnostics: Mapping[str, Any] | None = None,
        module: Any | None = None,
        val_dataloader: Any | None = None,
        task_weights: Mapping[str, float] | None = None,
        max_batches: int = 200,
        bins: int = 40,
        yscale: str = "linear",
        title: str | None = None,
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
        score = np.asarray(diag.get(self.score_key), dtype=np.float32)
        truth = np.asarray(diag.get(self.truth_key), dtype=np.float32)
        m = min(int(score.size), int(truth.size))
        if m <= 0:
            return None
        score = score[:m]
        truth = truth[:m]
        valid = np.isfinite(score) & np.isfinite(truth)
        score = score[valid]
        truth = truth[valid]
        if int(score.size) <= 0:
            return None

        y0 = score[truth < 0.5]
        y1 = score[truth >= 0.5]
        all_vals = np.concatenate([arr for arr in (y0, y1) if int(arr.size) > 0], axis=0)
        if int(all_vals.size) <= 0:
            return None
        lo = float(np.min(all_vals))
        hi = float(np.max(all_vals))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if hi <= lo:
            delta = 0.5 if abs(lo) < 1e-6 else 0.05 * abs(lo)
            lo -= delta
            hi += delta
        bin_edges = np.linspace(lo, hi, max(2, int(bins) + 1))

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        if int(y0.size) > 0:
            ax.hist(y0, bins=bin_edges, alpha=0.6, label=f"truth=0 (n={int(y0.size)})")
        if int(y1.size) > 0:
            ax.hist(y1, bins=bin_edges, alpha=0.6, label=f"truth=1 (n={int(y1.size)})")
        ax.set_title((title or self.title_default) + f"\nN={int(score.size)}")
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        if str(yscale).strip().lower() == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        return _save_and_show(fig=fig, save_path=save_path, show=show)

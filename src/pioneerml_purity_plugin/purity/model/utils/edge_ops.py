from __future__ import annotations

import torch
from torch_geometric.utils import dense_to_sparse


def fully_connected_edge_index_batch(batch: torch.Tensor) -> torch.Tensor:
    """Build complete directed edges per graph in a batched node vector (no self-loops)."""
    if int(batch.numel()) == 0:
        return batch.new_zeros((2, 0))
    adj = batch.unsqueeze(1) == batch.unsqueeze(0)
    n = int(adj.size(0))
    diag_idx = torch.arange(n, device=batch.device)
    adj[diag_idx, diag_idx] = False
    edge_index, _ = dense_to_sparse(adj)
    return edge_index


def build_purity_edge_attr(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Rebuild Omar-style 11D edge attributes from node features and edge connectivity.

    Output features:
    [dx, dy, dz, dE, dt, xz_xz, yz_yz, xz_yz, calo_calo, xz_calo, yz_calo]
    """
    if int(edge_index.numel()) == 0:
        return x.new_zeros((0, 11))

    # Keep local literals for TorchScript friendliness.
    norm_pos_atar = 10.0
    norm_e_atar = 1.0
    norm_pos_lyso = 100.0
    norm_e_lyso = 70.0
    norm_t = 500.0

    src, dst = edge_index[0], edge_index[1]
    u, v = x[src], x[dst]

    u_mod = torch.where(u[:, 7] > 0.5, 2, torch.where(u[:, 6] > 0.5, 1, 0))
    v_mod = torch.where(v[:, 7] > 0.5, 2, torch.where(v[:, 6] > 0.5, 1, 0))

    u_is_atar = (u_mod < 2).to(dtype=u.dtype)
    v_is_atar = (v_mod < 2).to(dtype=v.dtype)

    u_pos_scale = (u_is_atar.unsqueeze(1) * norm_pos_atar) + ((1.0 - u_is_atar).unsqueeze(1) * norm_pos_lyso)
    v_pos_scale = (v_is_atar.unsqueeze(1) * norm_pos_atar) + ((1.0 - v_is_atar).unsqueeze(1) * norm_pos_lyso)
    u_e_scale = (u_is_atar * norm_e_atar) + ((1.0 - u_is_atar) * norm_e_lyso)
    v_e_scale = (v_is_atar * norm_e_atar) + ((1.0 - v_is_atar) * norm_e_lyso)

    u_xyz = u[:, :3] * u_pos_scale
    v_xyz = v[:, :3] * v_pos_scale
    u_e = u[:, 3] * u_e_scale
    v_e = v[:, 3] * v_e_scale
    u_t = u[:, 4] * norm_t
    v_t = v[:, 4] * norm_t

    is_pure_atar_edge = ((u_mod < 2) & (v_mod < 2)).to(dtype=u.dtype)
    spatial_scale = (is_pure_atar_edge.unsqueeze(1) * norm_pos_atar) + (
        (1.0 - is_pure_atar_edge).unsqueeze(1) * norm_pos_lyso
    )
    energy_scale = (is_pure_atar_edge * norm_e_atar) + ((1.0 - is_pure_atar_edge) * norm_e_lyso)

    diff_xyz = (v_xyz - u_xyz) / spatial_scale
    diff_e = ((v_e - u_e) / energy_scale).unsqueeze(1)
    diff_t = ((v_t - u_t) / norm_t).unsqueeze(1)
    diffs = torch.cat([diff_xyz, diff_e, diff_t], dim=1)

    m_xz_xz = (u_mod == 0) & (v_mod == 0)
    m_yz_yz = (u_mod == 1) & (v_mod == 1)
    m_xz_yz = ((u_mod == 0) & (v_mod == 1)) | ((u_mod == 1) & (v_mod == 0))
    m_calo_calo = (u_mod == 2) & (v_mod == 2)
    m_xz_calo = ((u_mod == 0) & (v_mod == 2)) | ((u_mod == 2) & (v_mod == 0))
    m_yz_calo = ((u_mod == 1) & (v_mod == 2)) | ((u_mod == 2) & (v_mod == 1))

    out = x.new_zeros((diffs.shape[0], 11))
    out[:, :5] = diffs

    out[m_xz_xz, 1] = 0.0
    out[m_xz_xz, 5] = 1.0

    out[m_yz_yz, 0] = 0.0
    out[m_yz_yz, 6] = 1.0

    out[m_xz_yz, 0] = 0.0
    out[m_xz_yz, 1] = 0.0
    out[m_xz_yz, 7] = 1.0

    out[m_calo_calo, 8] = 1.0

    out[m_xz_calo, 3] = 0.0
    out[m_xz_calo, 9] = 1.0

    out[m_yz_calo, 3] = 0.0
    out[m_yz_calo, 10] = 1.0
    return out

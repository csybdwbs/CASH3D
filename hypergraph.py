from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HypergraphBundle:
    incidence_h: np.ndarray
    propagation_g: np.ndarray
    color_scalar: np.ndarray
    distance_matrix: np.ndarray


def _pairwise_sq_dist(x: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    return np.clip(x2 + x2.T - 2.0 * (x @ x.T), 0.0, None)


def _normalize_rgb(color_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(color_rgb, dtype=np.float32)
    if rgb.ndim != 2:
        raise ValueError(f"color_rgb must be 2D, got {rgb.shape}")
    if rgb.shape[1] < 3:
        rgb = np.repeat(rgb[:, :1], 3, axis=1)
    elif rgb.shape[1] > 3:
        rgb = rgb[:, :3]
    return rgb


def _inverse_distance_adjacency(distance_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    adjacency = 1.0 / (distance_matrix + float(eps))
    np.fill_diagonal(adjacency, 1.0)
    return adjacency.astype(np.float32)


def _build_knn_neighbors(distance_matrix: np.ndarray, k: int) -> list[np.ndarray]:
    n = distance_matrix.shape[0]
    k_eff = int(max(1, min(k, n)))
    neighbors = []
    for i in range(n):
        idx = np.argsort(distance_matrix[i])[:k_eff]
        if idx[0] != i:
            idx[0] = i
        neighbors.append(idx.astype(np.int64))
    return neighbors


def build_shared_color_aware_hypergraph(
    coords_3d: np.ndarray,
    color_rgb: np.ndarray,
    knn_k: int = 9,
    color_lambda: float = 1.0,
    adjacency_eps: float = 1e-6,
) -> HypergraphBundle:
    """
    Build a shared color-aware 3D hypergraph.

    d_ij = sqrt( ||(x_i,y_i,z_i)-(x_j,y_j,z_j)||_2^2
                 + lambda_c * ||rgb_i-rgb_j||_2^2 )
    A_ij is inverse to d_ij.
    For each center spot i, top-k similar spots form one hyperedge.
    """
    xyz = np.asarray(coords_3d, dtype=np.float32)
    rgb = _normalize_rgb(color_rgb)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"coords_3d must be (N,3), got {xyz.shape}")
    if not (0.0 <= float(color_lambda) <= 1.0):
        raise ValueError(f"color_lambda must be in [0, 1], got {color_lambda}.")

    xyz_sq = _pairwise_sq_dist(xyz)
    rgb_sq = _pairwise_sq_dist(rgb)
    distance_sq = xyz_sq + float(color_lambda) * rgb_sq
    distance = np.sqrt(np.clip(distance_sq, 0.0, None)).astype(np.float32)
    adjacency = _inverse_distance_adjacency(distance, eps=adjacency_eps)

    neighbors = _build_knn_neighbors(distance, k=knn_k)
    n = xyz.shape[0]
    h = np.zeros((n, n), dtype=np.float32)

    for center in range(n):
        idx = neighbors[center]
        h[idx, center] = adjacency[idx, center]

    # Hypergraph propagation matrix:
    # G = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
    d_v = np.sum(h, axis=1)
    d_e = np.sum(h, axis=0)
    d_v_inv_sqrt = np.diag(1.0 / np.sqrt(d_v + 1e-8))
    d_e_inv = np.diag(1.0 / (d_e + 1e-8))
    g = d_v_inv_sqrt @ h @ d_e_inv @ h.T @ d_v_inv_sqrt

    return HypergraphBundle(
        incidence_h=h.astype(np.float32),
        propagation_g=g.astype(np.float32),
        # Kept for downstream compatibility only.
        color_scalar=np.mean(rgb, axis=1, keepdims=True).astype(np.float32),
        distance_matrix=distance.astype(np.float32),
    )

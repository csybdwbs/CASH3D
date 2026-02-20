from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import anndata as ad
    import scanpy as sc
except Exception as exc:  # pragma: no cover - environment dependent
    ad = None
    sc = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_anndata() -> None:
    if ad is None or sc is None:
        raise ImportError(
            "scanpy/anndata are required for CASH3D alignment. "
            f"Import error: {_IMPORT_ERROR}"
        )


def _pairwise_dist(x: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.clip(x2 + x2.T - 2.0 * (x @ x.T), 0.0, None)
    return np.sqrt(d2).astype(np.float32)


def _align_with_paste(
    adata_list: list["ad.AnnData"],
    coor_key: str = "spatial",
    alpha: float = 0.1,
) -> list["ad.AnnData"]:
    try:
        import paste as pst
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PASTE is required for alignment_method='paste'. Install package 'paste-bio'."
        ) from exc

    if len(adata_list) < 2:
        raise ValueError("PASTE alignment requires at least 2 slices.")

    pairwise_maps = []
    for i in range(len(adata_list) - 1):
        pi = pst.pairwise_align(adata_list[i], adata_list[i + 1], alpha=alpha)
        pairwise_maps.append(pi)

    stacked = pst.stack_slices_pairwise(adata_list, pairwise_maps)
    out = [x.copy() for x in stacked]
    for adata_i in out:
        if coor_key in adata_i.obsm:
            adata_i.obsm["spatial_aligned"] = np.asarray(adata_i.obsm[coor_key]).copy()
    return out


def align_slices(
    adata_list: list["ad.AnnData"],
    alignment_method: str = "paste",
    coor_key: str = "spatial",
) -> list["ad.AnnData"]:
    method = alignment_method.lower()
    if method == "none":
        out = [x.copy() for x in adata_list]
        for adata_i in out:
            if coor_key in adata_i.obsm:
                adata_i.obsm["spatial_aligned"] = np.asarray(adata_i.obsm[coor_key]).copy()
        return out
    if method == "paste":
        return _align_with_paste(adata_list, coor_key=coor_key)
    raise ValueError(f"Unknown alignment_method: {alignment_method}")


def merge_aligned_slices_with_3d_coords(
    aligned_slices: list["ad.AnnData"],
    aligned_xy_key: str = "spatial_aligned",
    slice_key: str = "slice",
    coords_key: str = "3D_coor",
    slice_dist_micron: Optional[list[float]] = None,
    c2c_dist: float = 200.0,
) -> "ad.AnnData":
    _require_anndata()
    if not aligned_slices:
        raise ValueError("aligned_slices is empty.")

    merged = None
    xy_parts = []
    for idx, adata_i in enumerate(aligned_slices):
        if aligned_xy_key not in adata_i.obsm:
            raise ValueError(f"Slice {idx} missing '{aligned_xy_key}' in obsm.")
        cur = adata_i.copy()
        cur.var_names_make_unique()
        cur.obs_names = cur.obs_names.astype(str) + f"-slice{idx}"
        cur.obs[slice_key] = idx

        if merged is None:
            merged = cur
        else:
            shared_genes = merged.var_names.intersection(cur.var_names)
            merged = ad.concat(
                [merged[:, shared_genes], cur[:, shared_genes]],
                merge="same",
                join="inner",
                index_unique=None,
            )
        xy_parts.append(np.asarray(adata_i.obsm[aligned_xy_key], dtype=np.float32))

    merged.obs[slice_key] = merged.obs[slice_key].astype(int)
    xy = np.concatenate(xy_parts, axis=0).astype(np.float32)

    ref_xy = xy_parts[0]
    ref_dist = _pairwise_dist(ref_xy)
    uniq = np.sort(np.unique(ref_dist), axis=None)
    min_dist_ref = float(uniq[1]) if uniq.shape[0] > 1 else 1.0

    n_slices = len(aligned_slices)
    if slice_dist_micron is None:
        slice_dist_micron = [100.0] * (n_slices - 1)
    if len(slice_dist_micron) != n_slices - 1:
        raise ValueError(
            "slice_dist_micron must have length n_slices - 1, "
            f"got {len(slice_dist_micron)} for n_slices={n_slices}."
        )

    z = np.zeros(merged.n_obs, dtype=np.float32)
    offset = 0
    for i, dist in enumerate(slice_dist_micron):
        offset += aligned_slices[i].n_obs
        z[offset:] += float(dist) * (min_dist_ref / float(c2c_dist))

    merged.obsm[coords_key] = np.concatenate([xy, z.reshape(-1, 1)], axis=1).astype(np.float32)
    merged.obsm[aligned_xy_key] = xy.astype(np.float32)
    return merged


def build_merged_adata_from_slices(
    slice_h5ad_paths: list[Path],
    alignment_method: str = "paste",
    coor_key: str = "spatial",
    aligned_xy_key: str = "spatial_aligned",
    slice_key: str = "slice",
    coords_key: str = "3D_coor",
    slice_dist_micron: Optional[list[float]] = None,
    c2c_dist: float = 200.0,
) -> "ad.AnnData":
    _require_anndata()
    if not slice_h5ad_paths:
        raise ValueError("slice_h5ad_paths is empty.")
    if len(slice_h5ad_paths) < 2:
        raise ValueError(
            f"At least 2 slices are required to build 3D coordinates, got {len(slice_h5ad_paths)}."
        )
    if alignment_method.lower() == "none":
        raise ValueError(
            "alignment_method='none' is not allowed for raw slices. "
            "Use alignment_method='paste' for multi-slice alignment."
        )

    adata_list = [sc.read_h5ad(str(p)) for p in slice_h5ad_paths]
    aligned = align_slices(adata_list, alignment_method=alignment_method, coor_key=coor_key)
    return merge_aligned_slices_with_3d_coords(
        aligned_slices=aligned,
        aligned_xy_key=aligned_xy_key,
        slice_key=slice_key,
        coords_key=coords_key,
        slice_dist_micron=slice_dist_micron,
        c2c_dist=c2c_dist,
    )

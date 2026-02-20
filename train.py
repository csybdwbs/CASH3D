from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

# Keep scanpy/numba import stable in restricted environments.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

try:
    import scanpy as sc
except Exception as exc:  # pragma: no cover - environment dependent
    sc = None
    _SCANPY_IMPORT_ERROR = exc
else:
    _SCANPY_IMPORT_ERROR = None

from config import TrainConfig
from data import (
    build_merged_adata_from_slices,
    ensure_color_fields,
    extract_virchow2_embeddings,
    load_cash3d_data,
)
from hypergraph import build_shared_color_aware_hypergraph
from model import Cash3DModel


def _bool_flag(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_scanpy() -> None:
    if sc is None:
        raise ImportError(
            "scanpy is required for training with AnnData inputs. "
            f"Import error: {_SCANPY_IMPORT_ERROR}"
        )


def _load_or_build_adata(
    cfg: TrainConfig,
    slice_h5ad_paths: list[Path],
    raw_coor_key: str = "spatial",
) -> "sc.AnnData":
    _require_scanpy()
    if slice_h5ad_paths:
        if len(slice_h5ad_paths) < 2:
            raise ValueError(
                f"--slice-h5ad expects at least 2 slices for 3D training, got {len(slice_h5ad_paths)}."
            )
        print(f"[Step 1] Loading {len(slice_h5ad_paths)} slices and running {cfg.alignment_method} alignment...")
        adata = build_merged_adata_from_slices(
            slice_h5ad_paths=slice_h5ad_paths,
            alignment_method=cfg.alignment_method,
            coor_key=raw_coor_key,
            aligned_xy_key=cfg.aligned_xy_key,
            slice_key=cfg.slice_key,
            coords_key=cfg.coords_key,
            slice_dist_micron=cfg.slice_dist_micron,
            c2c_dist=cfg.c2c_dist,
        )
        merged_path = cfg.out_dir / "aligned_merged.h5ad"
        adata.write(str(merged_path))
        print(f"[Step 1] Saved aligned merged data: {merged_path}")
        return adata

    if cfg.adata_path is None:
        raise ValueError("Either --adata-path or --slice-h5ad must be provided.")
    print(f"[Step 1] Loading preprocessed data: {cfg.adata_path}")
    adata = sc.read_h5ad(str(cfg.adata_path))

    if cfg.slice_key not in adata.obs:
        raise ValueError(
            f"adata_path mode requires obs['{cfg.slice_key}'] to indicate multiple slices."
        )
    n_slices = int(np.unique(np.asarray(adata.obs[cfg.slice_key])).shape[0])
    if n_slices < 2:
        raise ValueError(
            f"adata_path must contain at least 2 slices in obs['{cfg.slice_key}'], got {n_slices}."
        )
    if cfg.coords_key not in adata.obsm:
        raise ValueError(
            f"adata_path mode requires obsm['{cfg.coords_key}'] (3D coordinates)."
        )
    return adata


def _inject_slicewise_virchow2_embeddings(
    cfg: TrainConfig,
    adata_merged: "sc.AnnData",
    slice_h5ad_paths: list[Path],
) -> None:
    if not slice_h5ad_paths:
        return
    if not cfg.run_virchow2_extract:
        return
    if cfg.img_embed_npy is not None:
        return

    _require_scanpy()
    print("[Step 1.5] Extracting Virchow2 embeddings slice-by-slice...")
    per_slice_feats = []
    for idx, slice_path in enumerate(slice_h5ad_paths):
        adata_i = sc.read_h5ad(str(slice_path))
        feat_i = extract_virchow2_embeddings(
            adata_obj=adata_i,
            embed_key=cfg.img_embed_key,
            model_name=cfg.virchow2_model_name,
            image_key=cfg.virchow2_image_key,
            library_id=cfg.virchow2_library_id,
            coords_key=cfg.virchow2_coords_key,
            patch_size=cfg.virchow2_patch_size,
            batch_size=cfg.virchow2_batch_size,
            device=cfg.model_device,
        )
        print(f"[Step 1.5] Slice {idx}: {slice_path.name}, features={feat_i.shape}")
        per_slice_feats.append(feat_i)

    merged_feat = np.concatenate(per_slice_feats, axis=0).astype(np.float32)
    if merged_feat.shape[0] != adata_merged.n_obs:
        raise ValueError(
            "Virchow2 feature rows do not match merged AnnData spots: "
            f"{merged_feat.shape[0]} vs {adata_merged.n_obs}."
        )
    adata_merged.obsm[cfg.img_embed_key] = merged_feat
    print(
        f"[Step 1.5] Injected merged Virchow2 embeddings into obsm['{cfg.img_embed_key}'] "
        f"with shape={merged_feat.shape}."
    )


def run_train(cfg: TrainConfig, slice_h5ad_paths: list[Path], raw_coor_key: str) -> Path:
    cfg.ensure_paths()
    set_seed(cfg.seed)

    # Step 1: multi-slice PASTE alignment / 3D coordinates, or load prebuilt data.
    adata = _load_or_build_adata(cfg, slice_h5ad_paths=slice_h5ad_paths, raw_coor_key=raw_coor_key)
    _inject_slicewise_virchow2_embeddings(cfg, adata_merged=adata, slice_h5ad_paths=slice_h5ad_paths)

    # Step 2: build shared color-aware hypergraph (before foundation embeddings as requested).
    print("[Step 2] Building shared color-aware 3D hypergraph...")
    ensure_color_fields(adata, color_rgb_key=cfg.color_rgb_key, color_scalar_key=cfg.color_scalar_key)
    bundle = build_shared_color_aware_hypergraph(
        coords_3d=adata.obsm[cfg.coords_key],
        color_rgb=adata.obsm[cfg.color_rgb_key],
        knn_k=cfg.knn_k,
        color_lambda=cfg.color_lambda,
        adjacency_eps=cfg.adjacency_eps,
    )
    adata.obsm["hypergraph_H"] = bundle.incidence_h
    adata.obsm["hypergraph_G"] = bundle.propagation_g
    adata.obsm[cfg.color_scalar_key] = bundle.color_scalar
    print(
        "[Step 2] Hypergraph ready:",
        f"H={bundle.incidence_h.shape}, G={bundle.propagation_g.shape}, k={cfg.knn_k}",
    )

    # Step 3: inject/load foundation embeddings (scGPT + Virchow2) as node features.
    print("[Step 3] Loading foundation embeddings for node features...")
    cash_data = load_cash3d_data(
        adata_obj=adata,
        coords_key=cfg.coords_key,
        slice_key=cfg.slice_key,
        color_rgb_key=cfg.color_rgb_key,
        color_scalar_key=cfg.color_scalar_key,
        expr_embed_key=cfg.expr_embed_key,
        img_embed_key=cfg.img_embed_key,
        expr_embed_npy=cfg.expr_embed_npy,
        img_embed_npy=cfg.img_embed_npy,
        run_scgpt_extract=cfg.run_scgpt_extract,
        scgpt_model_dir=cfg.scgpt_model_dir,
        scgpt_gene_col=cfg.scgpt_gene_col,
        scgpt_batch_size=cfg.scgpt_batch_size,
        run_virchow2_extract=cfg.run_virchow2_extract,
        virchow2_model_name=cfg.virchow2_model_name,
        virchow2_image_key=cfg.virchow2_image_key,
        virchow2_library_id=cfg.virchow2_library_id,
        virchow2_coords_key=cfg.virchow2_coords_key,
        virchow2_patch_size=cfg.virchow2_patch_size,
        virchow2_batch_size=cfg.virchow2_batch_size,
        model_device=cfg.model_device,
        strict_foundation_embeddings=cfg.strict_foundation_embeddings,
        normalize_embeddings=cfg.normalize_embeddings,
    )
    print(
        "[Step 3] Feature sources:",
        f"expr={cash_data.expr_source} shape={cash_data.expr_feats.shape};",
        f"img={cash_data.img_source} shape={cash_data.img_feats.shape}",
    )

    # Step 4: dual HGNN training with MMD alignment + reconstruction.
    print("[Step 4] Training CASH3D...")
    if str(cfg.model_device).lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(cfg.model_device))
    expr_t = torch.from_numpy(cash_data.expr_feats).float().to(device)
    img_t = torch.from_numpy(cash_data.img_feats).float().to(device)
    g_t = torch.from_numpy(bundle.propagation_g).float().to(device)

    model = Cash3DModel(
        expr_dim=cash_data.expr_feats.shape[1],
        img_dim=cash_data.img_feats.shape[1],
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        mmd_sigma=cfg.mmd_sigma,
        fusion_init_alpha=cfg.fusion_init_alpha,
    ).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out = None
    for epoch in range(cfg.epochs):
        out = model(
            expr_feats=expr_t,
            img_feats=img_t,
            g=g_t,
            beta_recon=cfg.beta_recon,
        )
        loss = out["loss_total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == cfg.epochs - 1:
            print(
                f"epoch={epoch:04d} "
                f"total={out['loss_total'].item():.4f} "
                f"mmd={out['loss_contrast'].item():.4f} "
                f"recon_expr={out['loss_recon_expr'].item():.4f} "
                f"recon_img={out['loss_recon_img'].item():.4f} "
                f"alpha={out['alpha'].item():.4f}"
            )

    assert out is not None
    adata_path = cfg.out_dir / "adata_with_cash3d_outputs.h5ad"

    adata.obsm["hypergraph_H"] = bundle.incidence_h.astype(np.float32)
    adata.obsm["hypergraph_G"] = bundle.propagation_g.astype(np.float32)
    adata.obsm["cash3d_z_expr"] = out["z_expr"].detach().cpu().numpy().astype(np.float32)
    adata.obsm["cash3d_z_img"] = out["z_img"].detach().cpu().numpy().astype(np.float32)
    adata.obsm["cash3d_z_unified"] = out["z_unified"].detach().cpu().numpy().astype(np.float32)
    adata.write(str(adata_path))

    print(f"[Done] Saved unified embedding in: {adata_path}")
    return adata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CASH3D trainer")
    parser.add_argument("--adata-path", type=Path, default=None)
    parser.add_argument("--slice-h5ad", nargs="*", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cash3d_run"))
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--alignment-method",
        type=str,
        default="paste",
        choices=["paste", "none"],
    )
    parser.add_argument("--raw-coor-key", type=str, default="spatial")
    parser.add_argument("--aligned-xy-key", type=str, default="spatial_aligned")
    parser.add_argument("--coords-key", type=str, default="3D_coor")
    parser.add_argument("--slice-key", type=str, default="slice")
    parser.add_argument("--slice-dist-micron", nargs="*", type=float, default=None)
    parser.add_argument("--c2c-dist", type=float, default=200.0)

    parser.add_argument("--color-rgb-key", type=str, default="color_rgb")
    parser.add_argument("--color-scalar-key", type=str, default="color_c3")
    parser.add_argument("--knn-k", type=int, default=9)
    parser.add_argument(
        "--color-lambda",
        type=float,
        default=1.0,
        help="Color weighting factor in [0, 1] for hypergraph distance.",
    )
    parser.add_argument("--adjacency-eps", type=float, default=1e-6)

    parser.add_argument("--expr-embed-key", type=str, default="expr_embed_scgpt")
    parser.add_argument("--img-embed-key", type=str, default="img_embed_virchow2")
    parser.add_argument("--expr-embed-npy", type=Path, default=None)
    parser.add_argument("--img-embed-npy", type=Path, default=None)
    parser.add_argument(
        "--run-scgpt-extract",
        type=_bool_flag,
        default=True,
        help="Run scGPT embedding extraction if expression embeddings are missing.",
    )
    parser.add_argument("--scgpt-model-dir", type=Path, default=None)
    parser.add_argument("--scgpt-gene-col", type=str, default="gene_name")
    parser.add_argument("--scgpt-batch-size", type=int, default=64)
    parser.add_argument(
        "--run-virchow2-extract",
        type=_bool_flag,
        default=True,
        help="Run Virchow2 patch embedding extraction if image embeddings are missing.",
    )
    parser.add_argument("--virchow2-model-name", type=str, default="hf-hub:paige-ai/Virchow2")
    parser.add_argument("--virchow2-image-key", type=str, default="hires")
    parser.add_argument("--virchow2-library-id", type=str, default=None)
    parser.add_argument("--virchow2-coords-key", type=str, default="spatial")
    parser.add_argument("--virchow2-patch-size", type=int, default=256)
    parser.add_argument("--virchow2-batch-size", type=int, default=64)
    parser.add_argument(
        "--model-device",
        type=str,
        default="auto",
        help="Device used by embedding extractors and training (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--strict-foundation-embeddings",
        type=_bool_flag,
        default=True,
        help="If true, require scGPT/Virchow2 embeddings. If false, allow fallback features.",
    )
    parser.add_argument("--normalize-embeddings", type=_bool_flag, default=True)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adamax weight decay (L2), must be >= 0.",
    )
    parser.add_argument(
        "--beta-recon",
        type=float,
        default=5.0,
        help="Reconstruction weight beta in [5, 10].",
    )
    parser.add_argument(
        "--lambda-recon",
        type=float,
        default=None,
        help="Alias of --beta-recon for backward compatibility.",
    )
    parser.add_argument("--mmd-sigma", type=float, default=1.0)
    parser.add_argument("--fusion-init-alpha", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slice_h5ad = args.slice_h5ad or []
    if not slice_h5ad and args.adata_path is None:
        raise ValueError("Provide either --slice-h5ad (recommended) or --adata-path.")
    if slice_h5ad and len(slice_h5ad) < 2:
        raise ValueError("--slice-h5ad requires at least 2 slice files.")

    beta_recon = args.beta_recon
    if args.lambda_recon is not None:
        beta_recon = args.lambda_recon
    if not (5.0 <= float(beta_recon) <= 10.0):
        raise ValueError(
            f"beta_recon must be in [5, 10], got {beta_recon}."
        )
    if float(args.weight_decay) < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {args.weight_decay}.")
    if not (0.0 <= float(args.color_lambda) <= 1.0):
        raise ValueError(f"color_lambda must be in [0, 1], got {args.color_lambda}.")
    if not args.strict_foundation_embeddings:
        raise ValueError("strict_foundation_embeddings must be true to guarantee foundation-model usage.")
    if not args.run_scgpt_extract:
        raise ValueError("run_scgpt_extract must be true to guarantee scGPT feature extraction.")
    if not args.run_virchow2_extract and args.img_embed_npy is None:
        raise ValueError(
            "run_virchow2_extract must be true (or provide --img-embed-npy from a foundation model)."
        )

    cfg = TrainConfig(
        adata_path=args.adata_path,
        out_dir=args.out_dir,
        seed=args.seed,
        alignment_method=args.alignment_method,
        slice_dist_micron=args.slice_dist_micron,
        c2c_dist=args.c2c_dist,
        coords_key=args.coords_key,
        slice_key=args.slice_key,
        aligned_xy_key=args.aligned_xy_key,
        color_rgb_key=args.color_rgb_key,
        color_scalar_key=args.color_scalar_key,
        knn_k=args.knn_k,
        color_lambda=args.color_lambda,
        adjacency_eps=args.adjacency_eps,
        expr_embed_key=args.expr_embed_key,
        img_embed_key=args.img_embed_key,
        expr_embed_npy=args.expr_embed_npy,
        img_embed_npy=args.img_embed_npy,
        run_scgpt_extract=args.run_scgpt_extract,
        scgpt_model_dir=args.scgpt_model_dir,
        scgpt_gene_col=args.scgpt_gene_col,
        scgpt_batch_size=args.scgpt_batch_size,
        run_virchow2_extract=args.run_virchow2_extract,
        virchow2_model_name=args.virchow2_model_name,
        virchow2_image_key=args.virchow2_image_key,
        virchow2_library_id=args.virchow2_library_id,
        virchow2_coords_key=args.virchow2_coords_key,
        virchow2_patch_size=args.virchow2_patch_size,
        virchow2_batch_size=args.virchow2_batch_size,
        model_device=args.model_device,
        strict_foundation_embeddings=args.strict_foundation_embeddings,
        normalize_embeddings=args.normalize_embeddings,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta_recon=beta_recon,
        mmd_sigma=args.mmd_sigma,
        fusion_init_alpha=args.fusion_init_alpha,
    )
    output_h5ad = run_train(cfg, slice_h5ad_paths=slice_h5ad, raw_coor_key=args.raw_coor_key)
    print("[CASH3D] output h5ad:", output_h5ad)


if __name__ == "__main__":
    main()

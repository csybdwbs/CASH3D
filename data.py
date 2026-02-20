from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np

from alignment import build_merged_adata_from_slices


PUBLIC_DATASET_CATALOG: dict[str, dict[str, str]] = {
    "dlpfc": {
        "description": "Dorsolateral prefrontal cortex (DLPFC) benchmark.",
        "url": "https://github.com/LieberInstitute/spatialLIBD",
    },
    "embryonic_heart": {
        "description": "Human embryonic heart slices (4.5, 6.5, and 9 PCW).",
        "url": "https://data.mendeley.com/datasets/dgnysc3zn5/1",
    },
    "breast_cancer": {
        "description": "HER2-positive breast cancer spatial transcriptomics.",
        "url": "https://doi.org/10.5281/zenodo.4751624",
    },
}


@dataclass
class Cash3DData:
    adata: "object"
    expr_feats: np.ndarray
    img_feats: np.ndarray
    coords_3d: np.ndarray
    color_rgb: np.ndarray
    slice_labels: np.ndarray
    expr_source: str
    img_source: str


def get_public_dataset_catalog() -> dict[str, dict[str, str]]:
    return {k: dict(v) for k, v in PUBLIC_DATASET_CATALOG.items()}


def get_dataset_links(dataset_name: str) -> dict[str, str]:
    key = str(dataset_name).strip().lower()
    if key not in PUBLIC_DATASET_CATALOG:
        available = ", ".join(sorted(PUBLIC_DATASET_CATALOG))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    return dict(PUBLIC_DATASET_CATALOG[key])


def list_slice_h5ad_paths(dataset_dir: Path) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"dataset_dir not found: {root}")
    paths = sorted(root.glob("*.h5ad"))
    if len(paths) < 2:
        raise ValueError(
            f"Expected at least 2 slice h5ad files in {root}, found {len(paths)}."
        )
    return paths


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _as_float2d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={arr.shape}")
    return arr


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    return (x - mean) / std


def _compute_color_scalar(color_rgb: np.ndarray) -> np.ndarray:
    var = np.var(color_rgb, axis=0) + 1e-8
    w = var / np.sum(var)
    return (color_rgb @ w.reshape(-1, 1)).astype(np.float32)


def _inject_npy_embedding(adata_obj, out_key: str, npy_path: Optional[Path]) -> bool:
    if npy_path is None:
        return False
    arr = _as_float2d(f"embedding file '{npy_path}'", np.load(str(npy_path)))
    if arr.shape[0] != adata_obj.n_obs:
        raise ValueError(
            f"Embedding '{npy_path}' has {arr.shape[0]} rows, but adata has {adata_obj.n_obs} spots."
        )
    adata_obj.obsm[out_key] = arr
    return True


def ensure_foundation_embeddings(
    adata_obj,
    expr_embed_key: str,
    img_embed_key: str,
    expr_embed_npy: Optional[Path] = None,
    img_embed_npy: Optional[Path] = None,
    strict: bool = True,
) -> tuple[str, str]:
    expr_src = None
    img_src = None

    if expr_embed_key not in adata_obj.obsm and _inject_npy_embedding(adata_obj, expr_embed_key, expr_embed_npy):
        expr_src = f"file:{expr_embed_npy}"
    if img_embed_key not in adata_obj.obsm and _inject_npy_embedding(adata_obj, img_embed_key, img_embed_npy):
        img_src = f"file:{img_embed_npy}"

    if expr_embed_key in adata_obj.obsm:
        expr_src = expr_src or f"obsm:{expr_embed_key}"
    if img_embed_key in adata_obj.obsm:
        img_src = img_src or f"obsm:{img_embed_key}"

    if strict and expr_src is None:
        raise ValueError(
            f"Missing expression foundation embedding '{expr_embed_key}'. "
            "Provide adata.obsm key or --expr-embed-npy."
        )
    if strict and img_src is None:
        raise ValueError(
            f"Missing image foundation embedding '{img_embed_key}'. "
            "Provide adata.obsm key or --img-embed-npy."
        )

    return expr_src or "fallback", img_src or "fallback"


def ensure_color_fields(
    adata_obj,
    color_rgb_key: str = "color_rgb",
    color_scalar_key: str = "color_c3",
) -> None:
    if color_rgb_key not in adata_obj.obsm:
        if "color_feat" not in adata_obj.obsm:
            raise ValueError(
                f"Missing color field '{color_rgb_key}'. "
                "Expected adata.obsm['color_rgb'] or fallback adata.obsm['color_feat']."
            )
        c = _as_float2d("obsm['color_feat']", adata_obj.obsm["color_feat"])
        if c.shape[1] == 1:
            c = np.repeat(c[:, :1], 3, axis=1)
        elif c.shape[1] > 3:
            c = c[:, :3]
        adata_obj.obsm[color_rgb_key] = c.astype(np.float32)

    rgb = _as_float2d(f"obsm['{color_rgb_key}']", adata_obj.obsm[color_rgb_key])
    if rgb.shape[1] < 3:
        rgb = np.repeat(rgb[:, :1], 3, axis=1)
    elif rgb.shape[1] > 3:
        rgb = rgb[:, :3]
    adata_obj.obsm[color_rgb_key] = rgb.astype(np.float32)

    if color_scalar_key not in adata_obj.obsm:
        adata_obj.obsm[color_scalar_key] = _compute_color_scalar(adata_obj.obsm[color_rgb_key])


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Image must be HxWxC, got shape={arr.shape}")
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if np.max(arr) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _pick_library_id(adata_obj, library_id: Optional[str]) -> str:
    if "spatial" not in adata_obj.uns:
        raise ValueError("adata.uns['spatial'] is required for Virchow2 patch extraction.")
    libs = list(adata_obj.uns["spatial"].keys())
    if not libs:
        raise ValueError("adata.uns['spatial'] has no library entries.")
    if library_id is not None:
        if library_id not in adata_obj.uns["spatial"]:
            raise ValueError(f"library_id='{library_id}' not found in adata.uns['spatial'].")
        return library_id
    return libs[0]


def _load_spatial_image(adata_obj, image_key: str = "hires", library_id: Optional[str] = None):
    lib = _pick_library_id(adata_obj, library_id=library_id)
    lib_entry = adata_obj.uns["spatial"][lib]
    if "images" not in lib_entry or image_key not in lib_entry["images"]:
        keys = list(lib_entry.get("images", {}).keys())
        raise ValueError(
            f"Image key '{image_key}' not found for library '{lib}'. Available keys: {keys}"
        )
    image = _to_uint8_image(lib_entry["images"][image_key])
    scalefactors = lib_entry.get("scalefactors", {})
    return image, scalefactors, lib


def _infer_spot_pixels(adata_obj, coords_key: str, image_key: str, scalefactors: dict):
    if coords_key not in adata_obj.obsm:
        raise ValueError(f"Missing coords key '{coords_key}' in adata.obsm.")
    coords = np.asarray(adata_obj.obsm[coords_key], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Coordinates '{coords_key}' must be (N,2+), got {coords.shape}")

    if image_key == "fullres":
        scale = 1.0
    else:
        scale_key = f"tissue_{image_key}_scalef"
        scale = float(scalefactors.get(scale_key, 1.0))

    x = np.round(coords[:, 0] * scale).astype(np.int64)
    y = np.round(coords[:, 1] * scale).astype(np.int64)
    return x, y


def _extract_square_patch(image: np.ndarray, cx: int, cy: int, patch_size: int) -> np.ndarray:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")
    half = patch_size // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - image.shape[1])
    pad_b = max(0, y1 - image.shape[0])

    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        image = np.pad(image, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="reflect")
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t

    patch = image[y0:y1, x0:x1]
    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        raise RuntimeError(
            f"Patch extraction failed for center=({cx},{cy}), got {patch.shape[:2]} expected {patch_size}"
        )
    return patch


def _resolve_device(device: str) -> str:
    if device.lower() != "auto":
        return device
    try:
        import torch
    except Exception:  # pragma: no cover
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_model_output(output):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("torch is required for model output normalization.") from exc

    if torch.is_tensor(output):
        x = output
    elif isinstance(output, (tuple, list)):
        x = _normalize_model_output(output[0])
    elif isinstance(output, dict):
        for key in ("x_norm_clstoken", "x_cls", "features", "pooler_output", "last_hidden_state"):
            if key in output:
                x = _normalize_model_output(output[key])
                break
        else:
            first_val = next(iter(output.values()))
            x = _normalize_model_output(first_val)
    else:
        raise TypeError(f"Unsupported model output type: {type(output)}")

    if x.ndim == 3:
        x = x[:, 0, :]
    elif x.ndim > 2:
        x = torch.flatten(x, start_dim=1)
    return x


def extract_virchow2_embeddings(
    adata_obj,
    embed_key: str = "img_embed_virchow2",
    model_name: str = "hf-hub:paige-ai/Virchow2",
    image_key: str = "hires",
    library_id: Optional[str] = None,
    coords_key: str = "spatial",
    patch_size: int = 256,
    batch_size: int = 64,
    device: str = "auto",
) -> np.ndarray:
    try:
        import timm
        import torch
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Virchow2 extraction requires timm, torch, and pillow.") from exc

    image, scalefactors, used_lib = _load_spatial_image(
        adata_obj=adata_obj,
        image_key=image_key,
        library_id=library_id,
    )
    x_px, y_px = _infer_spot_pixels(
        adata_obj=adata_obj,
        coords_key=coords_key,
        image_key=image_key,
        scalefactors=scalefactors,
    )

    model_device = _resolve_device(device)
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    except TypeError:
        model = timm.create_model(model_name, pretrained=True)
    model.eval().to(model_device)
    data_cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    features = []
    with torch.no_grad():
        start = 0
        n = adata_obj.n_obs
        while start < n:
            end = min(start + int(batch_size), n)
            batch_tensors = []
            for i in range(start, end):
                patch = _extract_square_patch(
                    image=image,
                    cx=int(x_px[i]),
                    cy=int(y_px[i]),
                    patch_size=int(patch_size),
                )
                batch_tensors.append(transform(Image.fromarray(patch)))
            batch = torch.stack(batch_tensors, dim=0).to(model_device)
            out = model(batch)
            emb = _normalize_model_output(out)
            features.append(emb.detach().cpu().numpy().astype(np.float32))
            start = end

    feat = np.concatenate(features, axis=0).astype(np.float32)
    adata_obj.obsm[embed_key] = feat
    print(
        f"[Virchow2] library={used_lib}, image_key={image_key}, patch={patch_size}, "
        f"device={model_device}, shape={feat.shape}"
    )
    return feat


def _run_scgpt_embed_data(adata_obj, model_dir: Path, gene_col: str, batch_size: int):
    _prepare_scgpt_runtime()
    try:
        from scgpt.tasks import embed_data
        import scgpt.tasks.cell_emb as cell_emb
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "scGPT extraction requires scgpt package. Install scgpt or provide precomputed embeddings."
        ) from exc

    # Avoid multiprocessing DataLoader workers in restricted environments.
    orig_affinity = getattr(cell_emb.os, "sched_getaffinity", None)
    if orig_affinity is not None:
        cell_emb.os.sched_getaffinity = lambda _pid: set()

    work = adata_obj.copy()
    if gene_col not in work.var.columns:
        work.var[gene_col] = work.var_names.astype(str)
    if not work.var_names.is_unique:
        work.var_names_make_unique()

    signatures = [
        dict(
            adata_or_file=work,
            model_dir=str(model_dir),
            gene_col=gene_col,
            batch_size=batch_size,
            device="cpu",
            use_fast_transformer=False,
            return_new_adata=True,
        ),
        dict(
            adata=work,
            model_dir=str(model_dir),
            gene_col=gene_col,
            batch_size=batch_size,
            device="cpu",
            use_fast_transformer=False,
            return_new_adata=True,
        ),
        dict(
            adata_or_file=work,
            model_dir=str(model_dir),
            gene_col=gene_col,
            batch_size=batch_size,
            device="cpu",
            use_fast_transformer=False,
        ),
        dict(
            adata=work,
            model_dir=str(model_dir),
            gene_col=gene_col,
            batch_size=batch_size,
            device="cpu",
            use_fast_transformer=False,
        ),
    ]
    try:
        for kwargs in signatures:
            try:
                return embed_data(**kwargs)
            except TypeError:
                continue
        return embed_data(
            work,
            str(model_dir),
            gene_col=gene_col,
            batch_size=batch_size,
            device="cpu",
            use_fast_transformer=False,
            return_new_adata=True,
        )
    finally:
        if orig_affinity is not None:
            cell_emb.os.sched_getaffinity = orig_affinity


def _prepare_scgpt_runtime() -> None:
    # On some Linux setups the system libstdc++ is too old for scGPT deps.
    # Preloading the environment's libstdc++ avoids CXXABI mismatch.
    candidates = []
    env_override = os.environ.get("CASH3D_LIBSTDCXX")
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(Path(sys.prefix) / "lib" / "libstdc++.so.6")
    candidates.append(Path.home() / "miniconda3" / "lib" / "libstdc++.so.6")
    candidates.append(Path.home() / "conda" / "lib" / "libstdc++.so.6")

    for cand in candidates:
        try:
            if cand.exists():
                ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
                return
        except Exception:
            continue


def extract_scgpt_embeddings(
    adata_obj,
    embed_key: str = "expr_embed_scgpt",
    model_dir: Optional[Path] = None,
    gene_col: str = "gene_name",
    batch_size: int = 64,
) -> np.ndarray:
    if model_dir is None:
        raise ValueError("model_dir is required for scGPT extraction.")

    result = _run_scgpt_embed_data(
        adata_obj=adata_obj,
        model_dir=Path(model_dir),
        gene_col=gene_col,
        batch_size=int(batch_size),
    )

    candidates = [result, adata_obj] if hasattr(result, "obsm") else [adata_obj]
    for obj in candidates:
        for key in ("X_scGPT", "X_scgpt", "scgpt", "scGPT", "X_embed"):
            if hasattr(obj, "obsm") and key in obj.obsm:
                emb = _as_float2d(f"obsm['{key}']", obj.obsm[key]).astype(np.float32)
                adata_obj.obsm[embed_key] = emb
                print(f"[scGPT] loaded from obsm['{key}'], shape={emb.shape}")
                return emb

    if isinstance(result, np.ndarray):
        emb = _as_float2d("scgpt_result", result).astype(np.float32)
        adata_obj.obsm[embed_key] = emb
        print(f"[scGPT] loaded from return ndarray, shape={emb.shape}")
        return emb

    raise RuntimeError(
        "scGPT embedding extraction finished but no embedding matrix was found. "
        "Expected one of: obsm['X_scGPT'], obsm['X_scgpt'], obsm['scgpt'], obsm['scGPT']."
    )


def populate_foundation_embeddings(
    adata_obj,
    expr_embed_key: str,
    img_embed_key: str,
    run_scgpt_extract: bool = False,
    scgpt_model_dir: Optional[Path] = None,
    scgpt_gene_col: str = "gene_name",
    scgpt_batch_size: int = 64,
    run_virchow2_extract: bool = False,
    virchow2_model_name: str = "hf-hub:paige-ai/Virchow2",
    virchow2_image_key: str = "hires",
    virchow2_library_id: Optional[str] = None,
    virchow2_coords_key: str = "spatial",
    virchow2_patch_size: int = 256,
    virchow2_batch_size: int = 64,
    model_device: str = "auto",
) -> tuple[bool, bool]:
    expr_done = False
    img_done = False

    if run_scgpt_extract:
        extract_scgpt_embeddings(
            adata_obj=adata_obj,
            embed_key=expr_embed_key,
            model_dir=scgpt_model_dir,
            gene_col=scgpt_gene_col,
            batch_size=scgpt_batch_size,
        )
        expr_done = True

    if run_virchow2_extract:
        extract_virchow2_embeddings(
            adata_obj=adata_obj,
            embed_key=img_embed_key,
            model_name=virchow2_model_name,
            image_key=virchow2_image_key,
            library_id=virchow2_library_id,
            coords_key=virchow2_coords_key,
            patch_size=virchow2_patch_size,
            batch_size=virchow2_batch_size,
            device=model_device,
        )
        img_done = True

    return expr_done, img_done


def load_cash3d_data(
    adata_obj,
    coords_key: str = "3D_coor",
    slice_key: str = "slice",
    color_rgb_key: str = "color_rgb",
    color_scalar_key: str = "color_c3",
    expr_embed_key: str = "expr_embed_scgpt",
    img_embed_key: str = "img_embed_virchow2",
    expr_embed_npy: Optional[Path] = None,
    img_embed_npy: Optional[Path] = None,
    run_scgpt_extract: bool = False,
    scgpt_model_dir: Optional[Path] = None,
    scgpt_gene_col: str = "gene_name",
    scgpt_batch_size: int = 64,
    run_virchow2_extract: bool = False,
    virchow2_model_name: str = "hf-hub:paige-ai/Virchow2",
    virchow2_image_key: str = "hires",
    virchow2_library_id: Optional[str] = None,
    virchow2_coords_key: str = "spatial",
    virchow2_patch_size: int = 256,
    virchow2_batch_size: int = 64,
    model_device: str = "auto",
    strict_foundation_embeddings: bool = True,
    normalize_embeddings: bool = True,
) -> Cash3DData:
    if coords_key not in adata_obj.obsm:
        raise ValueError(
            f"Missing '{coords_key}' in adata.obsm. "
            "Build 3D coordinates first using multi-slice PASTE alignment."
        )

    ensure_color_fields(adata_obj, color_rgb_key=color_rgb_key, color_scalar_key=color_scalar_key)

    expr_extracted, img_extracted = populate_foundation_embeddings(
        adata_obj=adata_obj,
        expr_embed_key=expr_embed_key,
        img_embed_key=img_embed_key,
        run_scgpt_extract=run_scgpt_extract,
        scgpt_model_dir=scgpt_model_dir,
        scgpt_gene_col=scgpt_gene_col,
        scgpt_batch_size=scgpt_batch_size,
        run_virchow2_extract=run_virchow2_extract,
        virchow2_model_name=virchow2_model_name,
        virchow2_image_key=virchow2_image_key,
        virchow2_library_id=virchow2_library_id,
        virchow2_coords_key=virchow2_coords_key,
        virchow2_patch_size=virchow2_patch_size,
        virchow2_batch_size=virchow2_batch_size,
        model_device=model_device,
    )
    if expr_extracted:
        print(f"[Foundation] Extracted expression embeddings into obsm['{expr_embed_key}'].")
    if img_extracted:
        print(f"[Foundation] Extracted image embeddings into obsm['{img_embed_key}'].")

    expr_src, img_src = ensure_foundation_embeddings(
        adata_obj,
        expr_embed_key=expr_embed_key,
        img_embed_key=img_embed_key,
        expr_embed_npy=expr_embed_npy,
        img_embed_npy=img_embed_npy,
        strict=strict_foundation_embeddings,
    )

    expr_feats = (
        _as_float2d(f"obsm['{expr_embed_key}']", adata_obj.obsm[expr_embed_key])
        if expr_embed_key in adata_obj.obsm
        else _as_float2d("adata.X", _to_dense(adata_obj.X))
    )
    if expr_embed_key not in adata_obj.obsm:
        expr_src = "fallback:adata.X"

    img_feats = (
        _as_float2d(f"obsm['{img_embed_key}']", adata_obj.obsm[img_embed_key])
        if img_embed_key in adata_obj.obsm
        else _as_float2d(f"obsm['{color_scalar_key}']", adata_obj.obsm[color_scalar_key])
    )
    if img_embed_key not in adata_obj.obsm:
        img_src = f"fallback:obsm['{color_scalar_key}']"

    if normalize_embeddings:
        expr_feats = _zscore(expr_feats).astype(np.float32)
        img_feats = _zscore(img_feats).astype(np.float32)

    coords_3d = _as_float2d(f"obsm['{coords_key}']", adata_obj.obsm[coords_key]).astype(np.float32)
    color_rgb = _as_float2d(f"obsm['{color_rgb_key}']", adata_obj.obsm[color_rgb_key]).astype(np.float32)

    if slice_key in adata_obj.obs:
        slice_labels = np.asarray(adata_obj.obs[slice_key]).astype(int)
    else:
        slice_labels = np.zeros(adata_obj.n_obs, dtype=np.int64)

    return Cash3DData(
        adata=adata_obj,
        expr_feats=expr_feats.astype(np.float32),
        img_feats=img_feats.astype(np.float32),
        coords_3d=coords_3d,
        color_rgb=color_rgb,
        slice_labels=slice_labels.astype(np.int64),
        expr_source=expr_src,
        img_source=img_src,
    )


__all__ = [
    "Cash3DData",
    "build_merged_adata_from_slices",
    "ensure_color_fields",
    "load_cash3d_data",
    "extract_scgpt_embeddings",
    "extract_virchow2_embeddings",
    "populate_foundation_embeddings",
    "get_public_dataset_catalog",
    "get_dataset_links",
    "list_slice_h5ad_paths",
]

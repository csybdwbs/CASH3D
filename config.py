from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    # Input / output
    adata_path: Optional[Path] = None
    out_dir: Path = Path("outputs/cash3d_run")
    seed: int = 0

    # Slice alignment and 3D construction
    alignment_method: str = "paste"
    slice_dist_micron: Optional[list[float]] = None
    c2c_dist: float = 200.0
    coords_key: str = "3D_coor"
    slice_key: str = "slice"
    aligned_xy_key: str = "spatial_aligned"

    # Color fields for hypergraph construction
    color_rgb_key: str = "color_rgb"
    color_scalar_key: str = "color_c3"
    knn_k: int = 9
    color_lambda: float = 1.0
    adjacency_eps: float = 1e-6

    # Foundation embeddings (node features)
    expr_embed_key: str = "expr_embed_scgpt"
    img_embed_key: str = "img_embed_virchow2"
    expr_embed_npy: Optional[Path] = None
    img_embed_npy: Optional[Path] = None
    run_scgpt_extract: bool = True
    scgpt_model_dir: Optional[Path] = None
    scgpt_gene_col: str = "gene_name"
    scgpt_batch_size: int = 64
    run_virchow2_extract: bool = True
    virchow2_model_name: str = "hf-hub:paige-ai/Virchow2"
    virchow2_image_key: str = "hires"
    virchow2_library_id: Optional[str] = None
    virchow2_coords_key: str = "spatial"
    virchow2_patch_size: int = 256
    virchow2_batch_size: int = 64
    model_device: str = "auto"
    strict_foundation_embeddings: bool = True
    normalize_embeddings: bool = True

    # Model / optimization
    hidden_dim: int = 512
    latent_dim: int = 64
    dropout: float = 0.0
    epochs: int = 1200
    lr: float = 5e-4
    weight_decay: float = 0.0
    beta_recon: float = 5.0
    mmd_sigma: float = 1.0
    fusion_init_alpha: float = 0.5

    def ensure_paths(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

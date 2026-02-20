# CASH3D

## Installation

```bash
cd /path/to/CASH3D
conda env create -f environment.yml
conda activate cash3d
```

## Foundation Models

### scGPT
- Project + pretrained checkpoints:  
  `https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo`

### Virchow2
- Model page:  
  `https://huggingface.co/paige-ai/Virchow2`
- Example download:

```bash
huggingface-cli download paige-ai/Virchow2 --local-dir /path/to/Virchow2
```

## Public Datasets (3)

- DLPFC: `https://github.com/LieberInstitute/spatialLIBD`
- Human embryonic heart: `https://data.mendeley.com/datasets/dgnysc3zn5/1`
- Breast cancer: `https://doi.org/10.5281/zenodo.4751624`

Dataset helper interface is in `data.py`:
- `get_public_dataset_catalog()`
- `get_dataset_links(dataset_name)`
- `list_slice_h5ad_paths(dataset_dir)`

```python
from pathlib import Path
from data import get_dataset_links, list_slice_h5ad_paths

print(get_dataset_links("breast_cancer"))
slice_paths = list_slice_h5ad_paths(Path("/path/to/breast_cancer_h5ad_slices"))
```

## Run (Multi-slice 3D)

```bash
python train.py \
  --slice-h5ad /path/slice1.h5ad /path/slice2.h5ad /path/slice3.h5ad \
  --alignment-method paste \
  --slice-dist-micron 10 300 \
  --knn-k 9 \
  --run-scgpt-extract true \
  --scgpt-model-dir /path/to/scgpt_checkpoint_dir \
  --run-virchow2-extract true \
  --virchow2-model-name hf-hub:paige-ai/Virchow2 \
  --virchow2-patch-size 256 \
  --out-dir outputs/run_3d

### Required input fields per slice `.h5ad`
- `obsm["spatial"]`: spot XY coordinates
- `uns["spatial"][library_id]["images"]["hires"]`: histology image
- `uns["spatial"][library_id]["scalefactors"]`: image scale metadata

### Optional: run with precomputed embeddings
If you already have embeddings, write them into `obsm` or pass `.npy` files:

```bash
python train.py \
  --slice-h5ad /path/slice1.h5ad /path/slice2.h5ad /path/slice3.h5ad \
  --alignment-method paste \
  --expr-embed-npy /path/expr_embed.npy \
  --img-embed-npy /path/img_embed.npy \
  --strict-foundation-embeddings true \
  --out-dir outputs/run_3d
```

Notes:
- With `--slice-h5ad ... --run-virchow2-extract true`, image embeddings are extracted per slice and then merged automatically.
- With `--adata-path ...`, make sure your `.h5ad` already contains valid image metadata if you want to run Virchow2 extraction directly.

## Output

- Output file: `outputs/run_3d/adata_with_cash3d_outputs.h5ad`
- Unified embedding: `obsm["cash3d_z_unified"]`

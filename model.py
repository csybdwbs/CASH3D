from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class HypergraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # Message propagation over shared hypergraph.
        return self.linear(torch.matmul(g, x))


class HGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = HypergraphConv(in_dim, hidden_dim)
        self.conv2 = HypergraphConv(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.conv1(x, g))
        h = self.dropout(h)
        z = self.conv2(h, g)
        return z


class ReconstructionHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Cash3DModel(nn.Module):
    """
    CASH3D core model:
      1) Dual HGNN encoders for expression/image hypergraphs (shared structure).
      2) MMD-based cross-modal alignment.
      3) Learnable fusion for unified embedding.
      4) Reconstruction heads for both modalities.
    """

    def __init__(
        self,
        expr_dim: int,
        img_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
        mmd_sigma: float = 1.0,
        fusion_init_alpha: float = 0.5,
    ):
        super().__init__()
        self.expr_encoder = HGNNEncoder(expr_dim, hidden_dim, latent_dim, dropout=dropout)
        self.img_encoder = HGNNEncoder(img_dim, hidden_dim, latent_dim, dropout=dropout)
        self.expr_recon_head = ReconstructionHead(latent_dim, hidden_dim, expr_dim)
        self.img_recon_head = ReconstructionHead(latent_dim, hidden_dim, img_dim)
        self.mmd_sigma = float(mmd_sigma)

        alpha0 = max(min(float(fusion_init_alpha), 1.0 - 1e-4), 1e-4)
        self.fusion_logit = nn.Parameter(torch.tensor(math.log(alpha0 / (1.0 - alpha0))))

    @staticmethod
    def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sq = torch.sum(x * x, dim=1, keepdim=True)
        y_sq = torch.sum(y * y, dim=1, keepdim=True).T
        return torch.clamp(x_sq + y_sq - 2.0 * torch.matmul(x, y.T), min=0.0)

    def mmd_loss(self, z_expr: torch.Tensor, z_img: torch.Tensor) -> torch.Tensor:
        sigma = max(self.mmd_sigma, 1e-6)
        gamma = 1.0 / (2.0 * sigma * sigma)
        d_xx = self._pairwise_sq_dist(z_expr, z_expr)
        d_yy = self._pairwise_sq_dist(z_img, z_img)
        d_xy = self._pairwise_sq_dist(z_expr, z_img)
        k_xx = torch.exp(-gamma * d_xx)
        k_yy = torch.exp(-gamma * d_yy)
        k_xy = torch.exp(-gamma * d_xy)
        return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    def forward(
        self,
        expr_feats: torch.Tensor,
        img_feats: torch.Tensor,
        g: torch.Tensor,
        beta_recon: float = 5.0,
    ) -> dict[str, torch.Tensor]:
        z_expr = self.expr_encoder(expr_feats, g)
        z_img = self.img_encoder(img_feats, g)

        alpha = torch.sigmoid(self.fusion_logit)
        z_unified = alpha * z_expr + (1.0 - alpha) * z_img

        expr_recon = self.expr_recon_head(z_unified)
        img_recon = self.img_recon_head(z_unified)

        l_contrast = self.mmd_loss(z_expr, z_img)
        l_recon_expr = torch.mean((expr_recon - expr_feats) ** 2)
        l_recon_img = torch.mean((img_recon - img_feats) ** 2)
        l_total = l_contrast + float(beta_recon) * (l_recon_expr + l_recon_img)

        return {
            "z_expr": z_expr,
            "z_img": z_img,
            "z_unified": z_unified,
            "alpha": alpha,
            "expr_recon": expr_recon,
            "img_recon": img_recon,
            "loss_total": l_total,
            "loss_contrast": l_contrast,
            "loss_recon_expr": l_recon_expr,
            "loss_recon_img": l_recon_img,
        }

# masked_autoencoder_spark.py

import torch
import torch.nn as nn
import math
from functools import partial
from .lib.models.spark import get_pose_net  
import matplotlib.pyplot as plt

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Create a 2D sine-cosine position embedding for images.
    Arguments:
        embed_dim (int): The dimension of the embedding.
        grid_size (int): The grid size of the input image (H = W).
    Returns:
        torch.Tensor: The position embeddings.
    """
    y_embed = torch.arange(grid_size).float()
    x_embed = torch.arange(grid_size).float()

    y_embed, x_embed = torch.meshgrid(y_embed, x_embed,indexing="ij")
    y_embed = y_embed.flatten()
    x_embed = x_embed.flatten()

    # Linear projection to [0, 2π]
    omega = torch.exp(
        torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
    )

    pos_x = x_embed.unsqueeze(1) * omega.unsqueeze(0)
    pos_y = y_embed.unsqueeze(1) * omega.unsqueeze(0)

    pos_embed = torch.cat([torch.sin(pos_x), torch.cos(pos_x), torch.sin(pos_y), torch.cos(pos_y)], dim=1)

    # If the embedding dimension is odd, drop the last dimension to match the input size
    if pos_embed.size(1) > embed_dim:
        pos_embed = pos_embed[:, :embed_dim]

    # Reshape the tensor to be 2D: [H*W, D]
    return pos_embed.reshape(grid_size * grid_size, embed_dim)
class MAESparK(nn.Module):
    def __init__(self, 
                 encoder_cfg, 
                 input_size=256, 
                 patch_size=16, 
                 embed_dim=256, 
                 masking_ratio=0.4):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio
        self.num_patches = (input_size // patch_size) ** 2

        # Encoder: use SparK backbone with output features only
        self.encoder = get_pose_net(encoder_cfg, is_train=True)
        self.encoder_head_removed = True

        # Projector: reduce encoder output to latent tokens
        self.proj = nn.Conv2d(256, embed_dim, kernel_size=1)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # Positional encoding
        self.pos_embed = get_2d_sincos_pos_embed(embed_dim, input_size // patch_size)

        # Decoder: CNN-based
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),  # 16→32
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # 32→64
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # 64→128
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),           # 128→256
        nn.Sigmoid(),
        )

    def patchify(self, imgs):
        """Split image into non-overlapping patches"""
        B, C, H, W = imgs.shape
        p = self.patch_size
        patches = imgs.reshape(B, C, H // p, p, W // p, p)
        patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, p * p)
        return patches

    def unpatchify(self, patches):
        """Reconstruct image from patches"""
        B, N, patch_dim = patches.shape
        p = self.patch_size
        h = w = int(self.input_size // p)
        patches = patches.reshape(B, h, w, p, p)
        img = patches.permute(0, 1, 3, 2, 4).reshape(B, 1, self.input_size, self.input_size)
        return img

    def forward(self, x, mask):
        """
        x: [B, 1, 256, 256]
        mask: [B, num_patches] — bool tensor (True if patch is masked)
        """
        latent = self.encoder(x)                         # [B, 256, H/16, W/16]
        latent = self.proj(latent)                       # [B, D, H/16, W/16]
        
        B, D, H, W = latent.shape
        latent = latent.permute(0, 2, 3, 1).reshape(B, -1, D)  # [B, N, D]

        # Prepare positional encodings
        pos = self.pos_embed.to(latent.device).unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        latent_with_pos = latent + pos

        # Expand mask to match shape
        mask = mask.bool()  # ensure it's bool type
        mask_token = self.mask_token.expand(B, self.num_patches, -1)  # [B, N, D]
        # Mix masked/unmasked tokens correctly
        full_tokens = torch.where(mask.unsqueeze(-1), mask_token, latent_with_pos)  # [B, N, D]

        # Reshape back to spatial for decoder
        x = full_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        # Decode
        recon = self.decoder(x)  # [B, 1, 256, 256]
        return recon

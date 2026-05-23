import os
import math
import numpy as np
import torch
from PIL import Image


def _normalize_per_channel(feat_np: np.ndarray) -> np.ndarray:
    """Normalize each channel to [0, 1]."""
    feat_min = feat_np.min(axis=(1, 2), keepdims=True)
    feat_max = feat_np.max(axis=(1, 2), keepdims=True)
    return (feat_np - feat_min) / (feat_max - feat_min + 1e-8)


def _apply_colormap(gray_img: np.ndarray, cmap_name: str = 'viridis') -> np.ndarray:
    """Apply matplotlib colormap to a grayscale image [0,1] -> RGB [0,255]."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    cmap = cm.get_cmap(cmap_name)
    # gray_img: (H, W) in [0, 1]
    rgba = cmap(gray_img)  # (H, W, 4)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def _create_grid(feat_np: np.ndarray, nrow: int = 8, pad: int = 2, 
                 cmap_name: str = 'viridis') -> np.ndarray:
    """Arrange feature channels into a color image grid.
    
    Args:
        feat_np (np.ndarray): Normalized array of shape (C, H, W) in [0, 1].
        nrow: Number of channels per row.
        pad: Padding between channels.
        cmap_name: Matplotlib colormap name.
    
    Returns:
        Grid image (H_grid, W_grid, 3) in uint8.
    """
    C, H, W = feat_np.shape
    ncol = math.ceil(C / nrow)
    
    # Each cell is H x W x 3 (RGB)
    grid_h = ncol * H + (ncol - 1) * pad
    grid_w = nrow * W + (nrow - 1) * pad
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for i in range(C):
        row = i % nrow
        col = i // nrow
        y = col * (H + pad)
        x = row * (W + pad)
        
        # Apply colormap to single channel
        ch_img = _apply_colormap(feat_np[i], cmap_name)
        grid[y:y+H, x:x+W] = ch_img
    
    return grid


def _create_overlay(feat_np: np.ndarray, cmap_name: str = 'turbo',
                    aggregation: str = 'max') -> np.ndarray:
    """Overlay all channels into a single image using max/mean aggregation.
    
    Args:
        feat_np (np.ndarray): Normalized array of shape (C, H, W) in [0, 1].
        cmap_name: Matplotlib colormap name.
        aggregation: 'max' or 'mean'.
    
    Returns:
        Overlay image (H, W, 3) in uint8.
    """
    if aggregation == 'max':
        aggregated = feat_np.max(axis=0)  # (H, W)
    else:
        aggregated = feat_np.mean(axis=0)   # (H, W)
    
    return _apply_colormap(aggregated, cmap_name)


def default_feature_visualizer(tensor: torch.Tensor, save_path: str, **kwargs):
    """Default feature visualization with grid or overlay mode.
    
    Args:
        tensor: Feature tensor of shape (B, C, H, W) or (C, H, W).
        save_path: Output image path.
        **kwargs:
            mode: 'grid' or 'overlay'. Default: 'grid'.
            max_channels: Max channels to visualize (grid mode only). Default: 64.
            nrow: Channels per row (grid mode only). Default: 8.
            pad: Padding between channels (grid mode only). Default: 2.
            cmap: Matplotlib colormap name. Default: 'viridis' (grid), 'turbo' (overlay).
            aggregation: 'max' or 'mean' for overlay mode. Default: 'max'.
    """
    mode = kwargs.get('mode', 'grid')
    max_channels = kwargs.get('max_channels', 64)
    nrow = kwargs.get('nrow', 8)
    pad = kwargs.get('pad', 2)
    cmap = kwargs.get('cmap', 'viridis' if mode == 'grid' else 'turbo')
    aggregation = kwargs.get('aggregation', 'max')

    # Handle batch dimension
    if tensor.dim() == 4:
        feat = tensor[0]
    elif tensor.dim() == 3:
        feat = tensor
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

    C, H, W = feat.shape
    feat_np = feat.detach().cpu().numpy()
    feat_np = _normalize_per_channel(feat_np)

    if mode == 'grid':
        if C > max_channels:
            feat_np = feat_np[:max_channels]
        
        grid = _create_grid(feat_np, nrow=nrow, pad=pad, cmap_name=cmap)
        Image.fromarray(grid).save(save_path)
        
    elif mode == 'overlay':
        overlay = _create_overlay(feat_np, cmap_name=cmap, aggregation=aggregation)
        Image.fromarray(overlay).save(save_path)
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'grid' or 'overlay'.")
    
def save_rgb_image(tensor: torch.Tensor, save_path: str, **kwargs):
    """Save a 3-channel tensor as RGB image directly.
    
    Args:
        tensor: Tensor of shape (3, H, W), values typically in [0, 1] or any range.
        save_path: Output image path.
        **kwargs:
            normalize: Whether to min-max normalize to [0, 1]. Default: True.
    """
    normalize = kwargs.get('normalize', True)
    
    assert tensor.shape[0] == 3, f"Expected 3 channels, got {tensor.shape[0]}"
    
    img_np = tensor.detach().cpu().numpy()  # (3, H, W)
    
    if normalize:
        img_min = img_np.min()
        img_max = img_np.max()
        if img_max - img_min > 1e-8:
            img_np = (img_np - img_min) / (img_max - img_min)
        else:
            img_np = np.zeros_like(img_np)
    else:
        # Assume already in [0, 1] or [0, 255], clip just in case
        img_np = np.clip(img_np, 0, 1)
    
    # (3, H, W) -> (H, W, 3) for PIL
    img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255).astype(np.uint8)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(img_np).save(save_path)
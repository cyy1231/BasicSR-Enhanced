import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img

import math
from tqdm import tqdm
from os import path as osp


@MODEL_REGISTRY.register()
class UCANModel(SRModel):
    """
    UCANModel with tile-based (patch-based) testing.
    
    Splits large-resolution images into smaller tiles to avoid OOM during inference,
    then merges outputs with overlapping boundaries to eliminate tile-edge artifacts.
    """
    def test(self):
        _, C, h, w = self.lq.size()
        scale = self.opt.get('scale', 1)
        
        # 1. Determine tile count (target ~200px per tile)
        n_h = h // 200 + 1
        n_w = w // 200 + 1

        # 2. Reflect-pad so that H/W are divisible by tile count
        mod_pad_h = (n_h - h % n_h) % n_h
        mod_pad_w = (n_w - w % n_w) % n_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')

        _, _, H, W = img.size()
        tile_h = H // n_h
        tile_w = W // n_w

        # 3. Build tile coordinates with 16px overlap (shave)
        shave_h = 16
        shave_w = 16

        def _make_slices(idx, n, tile_size, shave):
            base_start = idx * tile_size
            base_end = (idx + 1) * tile_size
            if n == 1:
                return base_start, base_end, 0, tile_size
            if idx == 0:
                return base_start, base_end + shave, 0, tile_size
            elif idx == n - 1:
                return base_start - shave, base_end, shave, tile_size + shave
            else:
                return base_start - shave, base_end + shave, shave, tile_size + shave

        tiles = []
        for i in range(n_h):
            for j in range(n_w):
                h0, h1, oh0, oh1 = _make_slices(i, n_h, tile_h, shave_h)
                w0, w1, ow0, ow1 = _make_slices(j, n_w, tile_w, shave_w)
                tiles.append({
                    'input_slice': (slice(h0, h1), slice(w0, w1)),
                    'output_slice': (slice(oh0, oh1), slice(ow0, ow1)),
                    'merge_top': slice(i * tile_h * scale, (i + 1) * tile_h * scale),
                    'merge_left': slice(j * tile_w * scale, (j + 1) * tile_w * scale),
                })

        img_chops = [img[..., t['input_slice'][0], t['input_slice'][1]] for t in tiles]

        target_net = getattr(self, 'net_g_ema', self.net_g)
        target_net.eval()

        # 4. Tile inference with per-tile feature visualization
        cfg = self.net_g_weight_visual_config
        do_visualize = cfg.get('visualize_layers') and cfg.get('visualize_during_test', False)
        net_name = cfg.get('name', 'net_g')
        vis_dir = osp.join(self.opt['path'].get('visualization', 'visualization'), 'features')
        prefix = f'iter_{getattr(self, "current_iter", "test")}'

        if do_visualize:
            self.setup_feature_hooks(
                target_net=target_net,
                net_name=net_name,
                layer_names=cfg.get('visualize_layers')
            )

        with torch.no_grad():
            outputs = []
            for idx, chop in enumerate(tqdm(img_chops, desc='Tile inference')):
                out = target_net(chop)
                outputs.append(out)
                
                # Save features immediately after each tile to avoid overwriting
                if do_visualize and hasattr(self, '_captured_features'):
                    tile_vis_dir = osp.join(vis_dir, f'tile_{idx}')
                    self.save_captured_features(tile_vis_dir, net_name=net_name, prefix=prefix)

        if do_visualize:
            self.remove_feature_hooks()

        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()

        # 5. Merge tile outputs (discard overlap bands)
        output = torch.zeros(1, C, H * scale, W * scale, device=img.device, dtype=img.dtype)
        for t, out in zip(tiles, outputs):
            out_crop = out[..., t['output_slice'][0], t['output_slice'][1]]
            output[..., t['merge_top'], t['merge_left']] = out_crop

        # 6. Crop padding to restore original size
        self.output = output[:, :, :h * scale, :w * scale]
       
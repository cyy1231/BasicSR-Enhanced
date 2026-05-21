import os

import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SwinIRModel(SRModel):

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        # base test start
        target_net = getattr(self, 'net_g_ema', self.net_g)
        target_net.eval()

        do_visualize = self.net_g_weight_visual_config.get('visualize_layers') and self.net_g_weight_visual_config.get('visualize_during_test', False)
        if do_visualize:
            self.setup_feature_hooks(
                target_net=target_net,
                net_name=self.net_g_weight_visual_config.get('name', 'net_g'),
                layer_names=self.net_g_weight_visual_config.get('visualize_layers')
            )

        with torch.no_grad():
            self.output = target_net(img)

        if do_visualize and hasattr(self, '_captured_features'):
            vis_dir = os.path.join(self.opt['path'].get('visualization', 'visualization'), 'features')
            prefix = f'iter_{getattr(self, "current_iter", "test")}'
            self.save_captured_features(
                vis_dir, 
                net_name=self.net_g_weight_visual_config.get('name', 'CATANet'),
                prefix=prefix
            )
            self.remove_feature_hooks()

        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()
        # end


        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

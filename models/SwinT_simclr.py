import torch.nn as nn
from models.swin_transformer import SwinTransformer



class SwinTSimCLR(nn.Module):

    def __init__(self):
        super(SwinTSimCLR, self).__init__()
        self.backbone = SwinTransformer(
                                        img_size=224, patch_size=4, in_chans=3, num_classes=128,
                                        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                        use_checkpoint=False)

    def forward(self, x):
        return self.backbone(x)
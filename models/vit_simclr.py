import torch.nn as nn
from vit_pytorch import ViT

from exceptions.exceptions import InvalidBackboneError


class ViTSimCLR(nn.Module):

    def __init__(self):
        super(ViTSimCLR, self).__init__()
        self.backbone = ViT(
                        image_size = 256,
                        patch_size = 32,
                        num_classes = 128,
                        dim = 1024,
                        depth = 6,
                        heads = 16,
                        mlp_dim = 512,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )
        # print('=' * 100)
        # print(self.backbone)
        # print('=' * 100)
        # dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

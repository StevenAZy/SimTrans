import os
import torch
import torch.nn as nn

from data_aug.ATLAS import ATLAS_N
from torchvision import transforms
from models.vit_simclr import ViTSimCLR
from models.SwinT_simclr import SwinTSimCLR
from models.resnet_simclr import ResNetSimCLR

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_dataset = ATLAS_N(Transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)





pretrain_resnet = torch.load('./runs/Jul03_21-49-02_SIAT-Station/checkpoint_1000.pth.tar')
pretrain_vit = torch.load('./runs/Jun28_16-14-42_SIAT-Station/checkpoint_1000.pth.tar')
resnet_model = ResNetSimCLR()
vit_model = ViTSimCLR()




SwinT_model = SwinTSimCLR()
pretrain_SwinT = torch.load('./runs/Jul02_16-09-01_SIAT-Station/checkpoint_1000.pth.tar')
# state_dict = pretrain_SwinT['state_dict']

model_dict =  SwinT_model.state_dict()
state_dict = {k:v for k,v in pretrain_SwinT.items() if k in model_dict.keys()}

model_dict.update(state_dict)
SwinT_model.load_state_dict(model_dict)
SwinT_model = nn.DataParallel(SwinT_model)
SwinT_model = SwinT_model.cuda()

for batch in train_loader:
    imgs = batch[0].cuda()
    labs = batch[1].cuda()
    preds = SwinT_model(imgs)

    # print(len(preds[0][1]))
    print(type(preds[0][1]))


# print(len(out))
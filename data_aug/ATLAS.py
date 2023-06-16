import os
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator


def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * size)),
                                            transforms.ToTensor()])
    return data_transforms



class ATLAS(Dataset):
    def __init__(self, n_views):
        self.n_views = n_views
        self.Transform = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(224), n_views)
        file_path = '/home/steven/code/SimTrans/ATLAS_1.2/Train_h5/train_nonormal.h5'
        train_file = h5py.File(file_path, 'r')
        self.image = train_file['image']
        self.label = train_file['label']

    def __getitem__(self, idx):
        image = Image.fromarray(self.image[idx]).convert('RGB')
        return [self.Transform(image) for i in range(self.n_views)]
        # return self.Transform(image)
        # return image

    def __len__(self):
        return len(self.image)


class ADE_Train(Dataset):
    def __init__(self, n_views):
        self.n_views = n_views
        self.Transform = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(96), n_views)
        self.train_img_path = '/home/steven/WorkSpace/Dataset/ade/ADEChallengeData2016/images/training/'
        self.train_lab_path = '/home/steven/WorkSpace/Dataset/ade/ADEChallengeData2016/annotations/training/'
        info_txt = '/home/steven/WorkSpace/Dataset/ade/ADEChallengeData2016/sceneCategories.txt'

        self.train = []

        f = open(info_txt, 'r')
        for line in f:
            line = line.rstrip()
            word = line.split()
            if 'train' in word[0]:
                self.train.append(word[0])

    def __getitem__(self, idx):
        image = Image.open(self.train_img_path + self.train[idx] + '.jpg').convert('RGB')
        # label = Image.open(self.train_lab_path + self.train[idx] + '.png')
        # if self.Transform:
        # print('=' * 100)
        # print(len(self.Transform(image)))
        return [self.Transform(image) for i in range(self.n_views)]
        # return self.Transform(image)
        # return image, label

    def __len__(self):
        return len(self.train)



class ATLAS_N(Dataset):
    def __init__(self, Transform = None):
        self.Transform = Transform
        file_path = '/home/steven/WorkSpace/Dataset/ATLAS_R1.2/Train_h5/train_nonormal.h5'
        train_file = h5py.File(file_path, 'r')
        self.image = train_file['image']
        self.label = train_file['label']

    def __getitem__(self, idx):
        image = Image.fromarray(self.image[idx]).convert('RGB')
        label = Image.fromarray(self.label[idx]).convert('RGB')
        if self.Transform:
            return self.Transform(image), self.Transform(label)
        return image, label

    def __len__(self):
        return len(self.image)
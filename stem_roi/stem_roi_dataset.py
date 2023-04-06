import numpy as np
import random
import os
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as tf
from torch.distributions.multivariate_normal import MultivariateNormal


class VimeoSepTuplet_QMap(Dataset):
    def __init__(self, data_root, is_training=True, cropsize=256, level_range=(0, 100), level=0):
        """
        Creates a Vimeo Septuplet object.
        MXH enhanced with qmap generation for variable-rate roi compression
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
        """
        self.data_root = data_root # .\vimeo_septuplet
        self.image_root = os.path.join(self.data_root, 'sequences') # .\vimeo_septuplet\sequences
        self.training = is_training
        self.cropsize = cropsize
        train_fn = os.path.join(self.data_root, 'vimeo_sep_trainlist_all.txt') # 91701
        # train_fn = os.path.join(self.data_root, 'sep_trainlist.txt') # 64612
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines() # ['00001/0001','00001/0002',...]
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.img_w = 448
        self.img_h = 256

        # todo: 不同的stage用不同大小的GOP来优化.是不是不同stage得是不同的数据集对象了？
        self.imglist1 = [1, 3, 5, 7]
        self.imglist2 = [1, 4, 7]
        self.imglist3 = [1, 7]


        self.level_range = level_range # 划分精度，默认为100
        self.level = level # for test dataloader
        self.p = 0.3
        self.grid = self._get_grid((self.cropsize, cropsize)) # for generate gaussian kernel


    def _get_crop_params(self):
        if self.img_w == self.cropsize and self.img_h == self.cropsize:
            return 0, 0, self.img_h, self.img_w

        if self.training:
            top = random.randint(0, self.img_h - self.cropsize)
            left = random.randint(0, self.img_w - self.cropsize)
        else:
            # center
            top = int(round((self.img_h - self.cropsize) / 2.))
            left = int(round((self.img_w - self.cropsize) / 2.))
        return top, left


    def _get_grid(self, size):
        x1 = torch.tensor(range(size[0]))
        x2 = torch.tensor(range(size[1]))
        grid_x1, grid_x2 = torch.meshgrid(x1, x2)

        grid1 = grid_x1.view(size[0], size[1], 1)
        grid2 = grid_x2.view(size[0], size[1], 1)
        grid = torch.cat([grid1, grid2], dim=-1)
        return grid


    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])

        # enhance dataset for learning different motion
        # rand = random.random()
        imgpaths = [imgpath + f'/f00{i}.png' for i in range(1, 8)]  # .\vimeo_septuplet\sequences\f001.png
        # if rand <= 0.25:
        #     imgpaths = [imgpath + f'/f00{i}.png' for i in self.imglist1]  # .\vimeo_septuplet\sequences\f001-3-5-7.png
        # elif rand <= 0.50:
        #     imgpaths = [imgpath + f'/f00{i}.png' for i in self.imglist2]  # .\vimeo_septuplet\sequences\f001-4-7.png
        # elif rand <= 0.75:
        #     imgpaths = [imgpath + f'/f00{i}.png' for i in self.imglist3]  # .\vimeo_septuplet\sequences\f001-7.png

        # Load images
        images = [Image.open(pth) for pth in imgpaths]
        top, left = self._get_crop_params()
        region = (left, top, left + self.cropsize, top + self.cropsize)
        images_ = []
        for image in images:
            image = image.crop(region)
            image = tf.to_tensor(image)
            images_.append(image)
        images = images_
        if random.random() >= 0.5:
            images = images[::-1]  # Data augmentation for reverse motion

        # todo: we can generate Qmap dataset in advance for faster training, every mode qmap in corresponding file.
        # todo: every time select mode and select random qmap from corresponding file
        # Generate QMap
        if self.training:
            qmap = np.zeros(images[0].size()[1:], dtype=float) # (cropsize,cropsize)
            sample = random.random()
            if sample < self.p:
                # uniform
                tmp = random.random()
                if tmp < 0.01:
                    qmap[:] = 0
                elif tmp < 0.20:
                    qmap[:] = (self.level_range[1] + 1) * (1-tmp) # 给高码率更多的训练机会，因为高码率训练收敛更慢
                else:
                    qmap[:] = (self.level_range[1] + 1) * random.random() # 不是很理解为什么要100+1，直接用100不行吗？因为random.random可能取不到1？
            elif sample < 2 * self.p:
                # gradation between two levels
                v1 = random.random() * self.level_range[1]
                v2 = random.random() * self.level_range[1]
                qmap = np.tile(np.linspace(v1, v2, self.cropsize), (self.cropsize, 1)).astype(float)
                if random.random() < 0.5:
                    qmap = qmap.T # 水平垂直翻转
            else:
                # gaussian kernel
                gaussian_num = int(1 + random.random() * 20)
                for i in range(gaussian_num):
                    mu_x = self.cropsize * random.random()
                    mu_y = self.cropsize * random.random()
                    var_x = 2000 * random.random() + 1000
                    var_y = 2000 * random.random() + 1000

                    m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
                    p = m.log_prob(self.grid)
                    kernel = torch.exp(p).numpy()
                    qmap += kernel
                qmap *= 100 / qmap.max() * (0.5 * random.random() + 0.5)
        else:
            qmap = np.zeros(images[0].size()[1:], dtype=float)  # (cropsize,cropsize)
            sample = random.random()
            # uniform
            qmap[:] = self.level

        qmap = torch.FloatTensor(qmap).unsqueeze(dim=0)
        qmap *= 1 / self.level_range[1]  # 0~100 -> 0~1

        return images, qmap

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)


def get_loader_roi(mode, data_root, batch_size, shuffle, num_workers, cropsize=256, level=0):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoSepTuplet_QMap(data_root, is_training=is_training, cropsize=cropsize, level=level)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
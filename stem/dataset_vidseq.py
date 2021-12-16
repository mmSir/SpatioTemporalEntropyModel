import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tf
from PIL import Image
import random

import torch


def TransformImgsWithSameCrop(images, cropsize):
    (i, j, h, w) = transforms.RandomCrop.get_params(images[0], (cropsize, cropsize))
    images_ = []
    for image in images:
        image = image.crop((j, i, j+w, i+h)) # top left corner (j,i); bottom right corner (j+w, i+h)
        image = tf.to_tensor(image)
        images_.append(image)

    images = images_

    return images


class VimeoSepTuplet(Dataset):
    def __init__(self, data_root, is_training, cropsize):
        """
        Creates a Vimeo Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = data_root # .\vimeo_septuplet
        self.image_root = os.path.join(self.data_root, 'sequences') # .\vimeo_septuplet\sequences
        self.training = is_training
        self.cropsize = cropsize
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt') # 64612
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines() # ['00001/0001','00001/0002',...]
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        if self.training:
            self.transforms = TransformImgsWithSameCrop
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

        self.imglist1 = [1, 3, 5, 7]
        self.imglist2 = [1, 4, 7]
        self.imglist3 = [1, 7]


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
        # Data augmentation
        if self.training:
            images = self.transforms(images, self.cropsize)
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]
        else:
            T = self.transforms
            images = [T(img_) for img_ in images]

        return images

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)


def get_loader(mode, data_root, batch_size, shuffle, num_workers, cropsize=256):
    if mode == 'train':
        is_training = True
    else:
        is_training = False
    dataset = VimeoSepTuplet(data_root, is_training=is_training, cropsize=cropsize)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.data_root = "D:\\MXH\\StartDeepLearning\\Dataset\\vimeo_septuplet"
    args.batch_size = 1
    args.test_batch_size = 1
    args.num_workers = 2

    train_loader = get_loader('train', args.data_root, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = get_loader('test', args.data_root, args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    # net = RDAutoEncoders(N=7, M=32)
    # testinput = torch.randn(1, 3, 7, 256, 256)  # (N, C, D, H, W)
    #
    # out = net(testinput)
    # print("test ok!")

    # test on cpu
    for i, images in enumerate(train_loader):
        # images = [img_.cuda() for img_ in images]
        images = torch.stack(images, dim=2)
        #print(images.size())

        # out = net(images)
        print(i)
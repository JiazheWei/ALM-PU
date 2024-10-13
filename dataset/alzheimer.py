import logging
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import ConcatDataset
import cv2
import os

from glob import glob
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


a_mean = (0.485, 0.456, 0.406)
a_std = (0.229, 0.224, 0.225)
train_dir = '/home/wjz/HolisticPU-main/dataset/Alzheimer_s Dataset/train'
test_dir='/home/wjz/HolisticPU-main/dataset/Alzheimer_s Dataset/test'


def get_alzheimer():
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(a_mean, a_std),
    ])
    train = datasets.ImageFolder(root=train_dir, transform=TrainTransform())
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    print(test_dataset.samples)

    for idx, (path, label) in enumerate(test_dataset.samples):
        # print('labelllllllll',label)
        if label == 2:
            test_dataset.samples[idx] = (path, 0)
        else:
            test_dataset.samples[idx] = (path, 1)

    # print('testttttttttttttttttttttt',test_dataset.samples.target)

    pos_num = 769
    num = 5121
    tot_list = np.arange(num)
    pos_list = np.arange(769,3329)


    label_list = np.random.choice(pos_list, 769, replace=False)
    np.random.shuffle(label_list)
    unlabeled_list = np.setdiff1d(tot_list, label_list)
    np.random.shuffle(unlabeled_list)
    train_labeled_dataset = Alzheimer(mode='labeled', indexs=label_list, root=train_dir, transform=TrainTransform())
    train_unlabeled_dataset = Alzheimer(mode='unlabeled', indexs=unlabeled_list, root=train_dir, transform=TrainTransform())

    print(len(train_labeled_dataset), len(train_unlabeled_dataset), len(test_dataset))
    # x=0
    # for data in train_labeled_dataset:
    #     (a,b),c=data
    #     if c == 0:
    #         x=x+1
    # print(x, len(train_labeled_dataset)-x)
    # x = 0
    # for data in train_unlabeled_dataset:
    #     (a, b), (d,c) = data
    #     if c ==0:
    #         x = x + 1
    # print(x, len(train_unlabeled_dataset) - x)
    # x=0
    # for data in test_dataset:
    #     a, c = data
    #     if c == 0:
    #         x = x + 1
    # print(x, len(test_dataset) - x)
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class Alzheimer(datasets.ImageFolder):
    def __init__(self, mode, indexs, root, transform):
        super(Alzheimer, self).__init__(root, transform)
        assert mode in ['labeled', 'unlabeled']
        self.mode = mode
        print('samplessssssss',len(self.samples))
        if mode == 'unlabeled':
            print('unlabeleddddddddd',np.max(indexs))
        if indexs is not None:
            data = []
            for i in indexs:
                data.append(self.samples[i])
                # if mode == 'labeled':
                #     print('samplesssssssss[indxe],,,',self.samples[i])
            self.samples = data

    def __getitem__(self, index):
        path, target = self.samples[index]
        # print('targetttttttttttttttt111111111',target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            print('tragettttttt22222222',target)
        if self.mode == 'unlabeled':
            if target == 2:
                target_t = 0
            else:
                target_t = target
            target_u = 1
            return sample, (target_u, target_t)
        elif self.mode == 'labeled':
            target = 0
            return sample, target

    def __len__(self):
        return len(self.samples)


class TrainTransform(object):
    def __init__(self, mean=a_mean, std=a_std):
        self.weak = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.strong = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=(1, 5), contrast=(1, 5), saturation=(1, 5), hue=(0.2, 0.4)),
            transforms.RandomRotation(degrees=(-30, 30)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong


if __name__ == '__main__':
    get_alzheimer()
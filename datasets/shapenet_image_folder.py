import json
import os
import pickle

import numpy as np
import torch
import h5py
import random
from PIL import Image
from skimage import io, transform
# from torch.utils.data.dataloader import default_collate

import utils.config as config
from datasets.base_dataset import BaseDataset


class ShapeNetImageFolder(BaseDataset):

    def __init__(self, folder, normalization, shapenet_options):
        super().__init__()
        self.normalization = normalization
        # self.resize_with_constant_border = shapenet_options.resize_with_constant_border
        self.file_list = []
        for fl in os.listdir(folder):
            file_path = os.path.join(folder, fl)
            # check image before hand
            try:
                if file_path.endswith(".gif"):
                    raise ValueError("gif's are results. Not acceptable")
                Image.open(file_path)
                self.file_list.append(file_path)
            except (IOError, ValueError):
                print("=> Ignoring %s because it's not a valid image" % file_path)

    def __getitem__(self, item):
        img_path = self.file_list[item]
        img = io.imread(img_path)

        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255

        # if self.resize_with_constant_border:
        #     img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE),
        #                            mode='constant', anti_aliasing=False)
        # else:
        img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        return {
            "images": img_normalized,
            "images_ori": img,
            "filename": self.file_list[item]
        }

    def __len__(self):
        return len(self.file_list)
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


# TODO: make pixel2mesh dataset
class ShapeNet(BaseDataset):
    def __init__(self, file_root, file_list_name, img_dir, mesh_pos, normalization, shapenet_options, logger):
        super(ShapeNet, self).__init__()
        self.file_root = file_root
        self.mesh_pos = mesh_pos
        self.img_dir = os.path.join(self.file_root, img_dir)
        self.file_names = []
        for lst in file_list_name:
            category_name = lst.split("_")[0]
            with open(os.path.join(self.file_root, "meta", lst), "r") as fp:
                for l in fp:
                    for r in range(1):
                        self.file_names.append((category_name, l.strip(), r))
        self.normalization = normalization
        self.shapenet_options = shapenet_options
        self.logger = logger

    def get_img(self, img_dir, num):
        img_h5 = os.path.join(img_dir, "%02d_sample.h5" % num)
        with h5py.File(img_h5, 'r') as h5_f:
            # the image data in h5 file is in 0-1
            img_ori = h5_f["img"][..., :3].astype(np.float32)

            points = h5_f["ptnm"][:, :3].astype(np.float32)
            normals = h5_f["ptnm"][:, 3:].astype(np.float32)
            norm_params = h5_f["norm_params"][:].astype(np.float32)
            camera_params = h5_f["camera_params"][:].astype(np.float32)
        points -= np.array(self.mesh_pos)
        assert points.shape[0] == normals.shape[0]
        return img_ori, points, normals, norm_params, camera_params

    def __getitem__(self, index):
        cat_id, obj, num = self.file_names[index]
        # read images pyramid
        img_dir = os.path.join(self.img_dir, cat_id, obj)
        h5_img, pt, nm, norm_params, camera_params = self.get_img(img_dir, num)
        tt = "/workspace/Users/wc/shapenet/ShapeNetRendering"
        real_img_dir = os.path.join(tt, cat_id, obj, "rendering", "%02d.png" % num )
        # import ipdb; ipdb.set_trace()
        img = io.imread(real_img_dir)
        if img.shape[2] > 3:  # has alpha channel
            img[np.where(img[:, :, 3] == 0)] = 255
        img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)
        # import ipdb; ipdb.set_trace()
        # to tensor
        img_ori = img
        img_ori = torch.from_numpy(np.transpose(img_ori, (2, 0, 1)))
        img_ori_normalized = self.normalize_img(img_ori) if self.normalization else img_ori
        # import ipdb; ipdb.set_trace()
        points = torch.from_numpy(pt)
        normals = torch.from_numpy(nm)
        norm_params = torch.from_numpy(norm_params)
        camera_params = torch.from_numpy(camera_params)
        filename = '_'.join(map(lambda x: str(x), [cat_id, obj, "%02d" % num]))


        return {
            "images": img_ori_normalized,
            # "images_ori": img_ori,
            # "points": points,
            # "normals": normals,
            # "norm_params": norm_params,
            # "camera_params": camera_params,
            "filename": filename
        }

    def __len__(self):
        return len(self.file_names)

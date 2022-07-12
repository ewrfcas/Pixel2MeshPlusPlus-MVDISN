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


class PoseShapeNet(BaseDataset):
    def __init__(self, file_root, file_list_name, img_dir, sdf_dir, normalization, shapenet_options, logger):
        super(PoseShapeNet, self).__init__()
        self.file_root = file_root
        self.sdf_dir = os.path.join(self.file_root, sdf_dir)
        self.img_dir = os.path.join(self.file_root, img_dir)
        # print(self.sdf_dir)
        # print(self.img_dir)
        self.file_names = []
        for lst in file_list_name:
            category_name = lst.split("_")[0]
            with open(os.path.join(self.file_root, "meta", lst), "r") as fp:
                for l in fp:
                    for r in range(24):
                        self.file_names.append((category_name, l.strip(), r))
        self.normalization = normalization
        self.shapenet_options = shapenet_options
        self.logger = logger

    def get_sdf_h5(self, sdf_h5_file, cat_id, obj):
        h5_f = h5py.File(sdf_h5_file, 'r')
        try:
            if ('pc_sdf_original' in h5_f.keys()
                    and 'pc_sdf_sample' in h5_f.keys()
                    and 'norm_params' in h5_f.keys()):
                ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
                sample_sdf = h5_f['pc_sdf_sample'][:-10000].astype(np.float32)
                ori_pt = ori_sdf[:, :3]  # , ori_sdf[:,3]
                ori_sdf_val = None
                if sample_sdf.shape[1] == 4:
                    sample_pt, sample_sdf_val = sample_sdf[:, :3], sample_sdf[:, 3:]
                else:
                    sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
                norm_params = h5_f['norm_params'][:]
                sdf_params = h5_f['sdf_params'][:]
            else:
                raise Exception(cat_id, obj, "no sdf and sample")
        finally:
            h5_f.close()
        return sample_pt, sample_sdf_val, norm_params, sdf_params

    def get_img(self, img_dir, num):
        img_h5 = os.path.join(img_dir, "%02d.h5" % num)
        cam_mat, cam_pos, trans_mat, obj_rot_mat, regress_mat = None, None, None, None, None
        with h5py.File(img_h5, 'r') as h5_f:
            trans_mat = h5_f["trans_mat"][:].astype(np.float32)
            obj_rot_mat = h5_f["obj_rot_mat"][:].astype(np.float32)
            regress_mat = h5_f["regress_mat"][:].astype(np.float32)
            cam_mat = h5_f["K"][:].astype(np.float32)
            cam_pos = h5_f["RT"][:].astype(np.float32)
            if self.shapenet_options.img_alpha:
                img_arr = h5_f["img_arr"][:].astype(np.float32)
                img_arr = img_arr / 255.
            else:
                img_raw = h5_f["img_arr"][:]
                img_arr = img_raw[:, :, :3]
                if self.shapenet_options.augcolorfore or self.shapenet_options.augcolorback:
                    r_aug = 60 * np.random.rand() - 30
                    g_aug = 60 * np.random.rand() - 30
                    b_aug = 60 * np.random.rand() - 30
                if self.shapenet_options.augcolorfore:
                    img_arr[img_raw[:, :, 3] != 0, 0] + r_aug
                    img_arr[img_raw[:, :, 3] != 0, 1] + g_aug
                    img_arr[img_raw[:, :, 3] != 0, 2] + b_aug
                if self.shapenet_options.augcolorback:
                    img_arr[img_raw[:, :, 3] == 0, 0] + r_aug
                    img_arr[img_raw[:, :, 3] == 0, 1] + g_aug
                    img_arr[img_raw[:, :, 3] == 0, 2] + b_aug
                else:
                    img_arr[img_raw[:, :, 3] == 0] = [255, 255, 255]
                img_arr = np.clip(img_arr, 0, 255)
                img_arr = img_arr.astype(np.float32) / 255.
            if self.shapenet_options.resize_with_constant_border:
                img = transform.resize(img_arr, (config.IMG_SIZE, config.IMG_SIZE),
                                       mode='constant', anti_aliasing=False).astype(np.float32)  # to match behavior of old versions
            else:
                img = transform.resize(img_arr, (config.IMG_SIZE, config.IMG_SIZE)).astype(np.float32)
        return img, cam_mat, cam_pos, trans_mat, obj_rot_mat, regress_mat

    def __getitem__(self, index):
        cat_id, obj, num = self.file_names[index]

        img_dir = os.path.join(self.img_dir, cat_id, obj)
        img_h5 = os.path.join(img_dir, "%02d_sample.h5" % num)
        """
        fp.create_dataset('img', data=im)
        # fp.create_dataset('norm_params', data=norm_para)
        fp.create_dataset('camera_params', data=param)
        fp.create_dataset('ptnm', data=train_data)
        # fp.create_dataset('sdf_params', data=sdf_params)
        fp.create_dataset('pc_sdf_original', data=pc_sdf_original)
        fp.create_dataset('pc_sdf_sample', data=pc_sdf_sample)
        # fp.create_dataset('img_arr', data=img_arr)
        # fp.create_dataset('trans_mat', data=trans_mat)
        # fp.create_dataset('K', data=K)
        # fp.create_dataset('RT', data=RT)
        # fp.create_dataset('obj_rot_mat', data=obj_rot_mat)
        # fp.create_dataset('regress_mat', data=regress_mat)

        """

        with h5py.File(img_h5, 'r') as h5_f:
            # img_arr
            img_raw = h5_f["img_arr"][:]
            img_arr = img_raw[:, :, :3]
            img_arr = np.clip(img_arr, 0, 255)
            img_arr = img_arr.astype(np.float32) / 255.
            img_arr = transform.resize(img_arr, (config.IMG_SIZE, config.IMG_SIZE)).astype(np.float32)
            # other
            trans_mat = h5_f["trans_mat"][:].astype(np.float32)
            obj_rot_mat = h5_f["obj_rot_mat"][:].astype(np.float32)
            regress_mat = h5_f["regress_mat"][:].astype(np.float32)
            K = h5_f["K"][:].astype(np.float32)
            RT = h5_f["RT"][:].astype(np.float32)
            ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
            sample_sdf = h5_f['pc_sdf_sample'][:-10000].astype(np.float32)
            sample_pt, sample_sdf_val = sample_sdf[:, :3], sample_sdf[:, 3:]
            norm_params = h5_f['norm_params'][:]
            sdf_params = h5_f['sdf_params'][:]

        norm_mat = self.get_norm_matrix(norm_params)

        sample_sdf_val = sample_sdf_val - 0.003  # the value of iso-surface 0.003
        sample_sdf_val = sample_sdf_val.T
        choice = np.asarray(random.sample(range(sample_pt.shape[0]), self.shapenet_options.num_points), dtype=np.int32)
        sample_pt_choice = sample_pt[choice, :]
        sample_sdf_val_choice = sample_sdf_val[:, choice]

        # to tensor
        img = torch.from_numpy(np.transpose(img_arr, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img
        sdf_points = torch.from_numpy(sample_pt_choice)
        sdf_value = torch.from_numpy(sample_sdf_val_choice)
        norm_params = torch.from_numpy(norm_params)
        sdf_params = torch.from_numpy(sdf_params)
        regress_mat = torch.from_numpy(regress_mat)
        trans_mat = torch.from_numpy(trans_mat)
        filename = '_'.join(map(lambda x: str(x), [cat_id, obj, num]))

        norm_mat = torch.from_numpy(norm_mat)

        if self.shapenet_options.rot:
            sample_pt_rot = np.dot(sample_pt[choice, :], obj_rot_mat)
        else:
            sample_pt_rot = sample_pt[choice, :]
        sdf_points_rot = torch.from_numpy(sample_pt_rot)

        K = torch.from_numpy(K)
        K = K.t()

        RT = torch.from_numpy(RT)

        rot_mat_inv = np.array([[1., 0.,  0., 0.],
                               [0., 0.,  1., 0.],
                               [0., -1., 0., 0.],
                               [0., 0.,  0., 1.]], dtype=np.float32)

        rot_mat_inv = torch.from_numpy(rot_mat_inv)

        return {
            "images": img_normalized,
            "sdf_points": sdf_points,
            "sdf_points_rot": sdf_points_rot,
            "sdf_value": sdf_value,
            "norm_params": norm_params,
            "sdf_params": sdf_params,
            "trans_mat": trans_mat,
            "regress_mat": regress_mat,
            "K": K,
            "RT": RT,
            "rot_mat_inv": rot_mat_inv,
            "norm_mat": norm_mat,
            "filename": filename
        }

    def __len__(self):
        return len(self.file_names)

    def get_norm_matrix(self, norm_params):
        center, m, = norm_params[:3], norm_params[3]
        x,y,z = center[0], center[1], center[2]
        M_inv = np.asarray(
            [[m, 0., 0., 0.],
            [0., m, 0., 0.],
            [0., 0., m, 0.],
            [0., 0., 0., 1.]]
        )
        T_inv = np.asarray(
            [[1.0 , 0., 0., 0.],
             [0., 1.0 , 0., 0.],
             [0., 0., 1.0 , 0.],
             [x, y, z, 1.]]
        )
        return np.matmul(M_inv, T_inv).astype(np.float32)
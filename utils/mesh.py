import os
import pickle

import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix

import utils.config as config


def torch_sparse_tensor(indices, value, size):
    coo = coo_matrix((value, (indices[:, 0], indices[:, 1])), shape=size)
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, shape)


"""
dat = [info['coords'],
       info['support']['stage1'],
       info['support']['stage2'],
       info['support']['stage3'],
       info['support']['stage4'],
       [info['unpool_idx']['stage1_2'],
        info['unpool_idx']['stage2_3'],
        info['unpool_idx']['stage3_4']
       ],
       [np.zeros((1,4), dtype=np.int32)]*4,
       [np.zeros((1,4))]*4,
       [info['lap_idx']['stage1'],
        info['lap_idx']['stage2'],
        info['lap_idx']['stage3'],
        info['lap_idx']['stage4']
       ],
      ]
"""


class Ellipsoid(object):

    def __init__(self, mesh_pos, file=config.ELLIPSOID_PATH, refinefile=config.REFINE_META_PATH):
        with open(file, "rb") as fp:
            fp_info = pickle.load(fp, encoding='latin1')

        # shape: n_pts * 3
        self.coord = torch.tensor(fp_info[0]) - torch.tensor(mesh_pos, dtype=torch.float)

        # edges & faces & lap_idx
        # edge: num_edges * 2
        # faces: num_faces * 4
        # laplace_idx: num_pts * 10
        self.edges, self.laplace_idx = [], []

        for i in range(3):
            self.edges.append(torch.tensor(fp_info[1 + i][1][0], dtype=torch.long))
            self.laplace_idx.append(torch.tensor(fp_info[7][i], dtype=torch.long))

        # unpool index
        # num_pool_edges * 2
        # pool_01: 462 * 2, pool_12: 1848 * 2
        self.unpool_idx = [torch.tensor(fp_info[4][i], dtype=torch.long) for i in range(2)]

        # loops and adjacent edges
        self.adj_mat = []
        for i in range(1, 4):
            # 0: np.array, 2D, pos
            # 1: np.array, 1D, vals
            # 2: tuple - shape, n * n
            adj_mat = torch_sparse_tensor(*fp_info[i][1])
            self.adj_mat.append(adj_mat)

        ellipsoid_dir = os.path.dirname(file)
        self.faces = []
        self.obj_fmt_faces = []
        # faces: f * 3, original ellipsoid, and two after deformations
        for i in range(1, 4):
            face_file = os.path.join(ellipsoid_dir, "face%d.obj" % i)
            faces = np.loadtxt(face_file, dtype='|S32')
            self.obj_fmt_faces.append(faces)
            self.faces.append(torch.tensor(faces[:, 1:].astype(np.int) - 1))

        with open(refinefile, "rb") as refine_fp:
            refine_fp_info = pickle.load(refine_fp, encoding='latin1')

        self.sample_coord = torch.FloatTensor(refine_fp_info['sample_coord'])
        self.sample_adj_mat = torch_sparse_tensor(*refine_fp_info['sample_cheb'][1]).to_dense()


def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3] * np.sin(phi)
    temp = param[3] * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])
    return cam_mat, cam_pos

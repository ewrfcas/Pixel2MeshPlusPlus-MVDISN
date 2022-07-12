import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers import PerceptualPooling, PointMLP, SDFDecoder


class PoseNet(nn.Module):
    def __init__(self, options):
        super(PoseNet, self).__init__()
        self.options = options
        self.globalfeat_dim = 1024
        self.nn_encoder = get_backbone(self.options)

        self.distratio = nn.Sequential(
            nn.Linear(self.globalfeat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        self.ortho6d = nn.Sequential(
            nn.Linear(self.globalfeat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6)
        )

    @staticmethod
    def get_fixed(batch_size, device):
        CAM_MAX_DIST = torch.tensor(1.75, dtype=torch.float32, device=device)
        CAM_MAX_DIST = torch.reshape(CAM_MAX_DIST, [1,1,1]).repeat([batch_size,1,1])

        CAM_ROT = torch.tensor(np.asarray([[0.0, 0.0, 1.0],
                                           [1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0]], dtype=np.float32), device=device)
        CAM_ROT = CAM_ROT.unsqueeze(0).repeat([batch_size,1,1])

        R_camfix = torch.tensor(np.asarray([[1., 0., 0.],
                                            [0., -1., 0.],
                                            [0., 0., -1.]], dtype=np.float32), device=device)
        R_camfix = R_camfix.unsqueeze(0).repeat([batch_size,1,1])

        return CAM_MAX_DIST, CAM_ROT, R_camfix

    @staticmethod
    def normalize_vector(v):
        batch= v.shape[0]
        v_mag = torch.sqrt(torch.sum(v * v, dim=1, keepdim=True))
        v_mag = torch.max(v_mag, torch.tensor(1e-8, device=v.device))
        v = v / v_mag
        return v

    def compute_rotation_matrix_from_ortho6d(self, poses):
        x_raw = poses[:, 0:3]#batch*3
        y_raw = poses[:, 3:6]#batch*3
        x = self.normalize_vector(x_raw) #batch*3
        z = torch.cross(x, y_raw, dim=1) #batch*3
        z = self.normalize_vector(z)#batch*3
        y = torch.cross(z, x, dim=1) #batch*3
        # print('x', x.shape, 'y', y.shape, 'z', z.shape)
        x = torch.reshape(x, [-1, 3, 1])
        y = torch.reshape(y, [-1, 3, 1])
        z = torch.reshape(z, [-1, 3, 1])
        matrix = torch.cat([x,y,z], 2) #batch*3*3
        # print('matrix', matrix.shape)
        return matrix

    def forward(self, input_batch):
        img = input_batch["images"]
        # import ipdb; ipdb.set_trace()

        batch_size = img.size(0)
        device = img.device
        img_featuremaps, img_feat_global = self.nn_encoder(img)
        #img_feat_local = self.percep_pooling(img_featuremaps, pc, trans_mat)

        # pred_rotation, pred_translation, pred_RT
        img_feat_global = img_feat_global.reshape(batch_size, -1)
        dist = self.distratio(img_feat_global)
        dist = torch.sigmoid(dist) * 0.4 + 0.6
        distance_ratio = torch.reshape(dist, (batch_size, 1, 1))

        rotation = self.ortho6d(img_feat_global)
        pred_rotation = torch.reshape(rotation, (batch_size, 6))

        CAM_MAX_DIST, R_obj2cam_inv, R_camfix = self.get_fixed(batch_size, device)

        cam_location_inv = torch.cat([distance_ratio * CAM_MAX_DIST, torch.zeros([batch_size, 1, 2], device=device)], 2)

        R_camfix_inv = R_camfix.permute((0, 2, 1))

        pred_translation_inv = cam_location_inv @ R_obj2cam_inv @ R_camfix_inv * -1.0

        pred_rotation_mat_inv = self.compute_rotation_matrix_from_ortho6d(pred_rotation)
        
        pred_RT_inv = torch.cat([pred_rotation_mat_inv, pred_translation_inv], 1)
        
        return {
            "pred_rotation": pred_rotation_mat_inv,
            "pred_translation": pred_translation_inv,
            "pred_RT": pred_RT_inv,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from plyfile import PlyData, PlyElement


def write_point_ply(filename, v, n):
    """Write point cloud to ply file.

    Args:
      filename: str, filename for ply file to load.
      v: np.array of shape [#v, 3], vertex coordinates
      n: np.array of shape [#v, 3], vertex normals
    """
    vn = np.concatenate([v, n], axis=1)
    vn = [tuple(vn[i]) for i in range(vn.shape[0])]
    vn = np.array(vn, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    el = PlyElement.describe(vn, 'vertex')
    PlyData([el]).write(filename)


class PerceptualPooling(nn.Module):
    def __init__(self, options):
        super(PerceptualPooling, self).__init__()
        self.options = options

    def forward(self, img_featuremaps, pc, trans_mat, sdf_value):
        x1, x2, x3, x4, x5 = img_featuremaps
        f1 = F.interpolate(x1, size=self.options.map_size, mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=self.options.map_size, mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=self.options.map_size, mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=self.options.map_size, mode='bilinear', align_corners=True)
        f5 = F.interpolate(x5, size=self.options.map_size, mode='bilinear', align_corners=True)
        # pc B*N*3
        homo_const = torch.ones((pc.shape[0], pc.shape[1], 1), device=pc.device, dtype=pc.dtype)
        homo_pc = torch.cat((pc, homo_const), dim=-1)
        pc_xyz = torch.matmul(homo_pc, trans_mat)  # pc_xyz B*N*3
        # pc_xyz = torch.matmul(trans_mat.transpose(1, 2), homo_pc.transpose(1, 2)).transpose(1, 2)
        # import ipdb
        # ipdb.set_trace()
        # tmp_xyz = pc_xyz[0, :, :].cpu().numpy()
        # sdf_value = sdf_value[0].cpu().numpy()
        # sdf_value = sdf_value.transpose()
        # tmp_xyz = tmp_xyz[np.abs(sdf_value) < 0.05, :]
        # tmp_normal = np.ones_like(tmp_xyz)
        # write_point_ply('tmp.ply', tmp_xyz, tmp_normal)

        # import ipdb
        # ipdb.set_trace()
        pc_xy = torch.div(pc_xyz[:, :, :2], (pc_xyz[:, :, 2:] + 1e-8))  # avoid divide zero
        pc_xy = torch.clamp(pc_xy, 0.0, self.options.map_size - 1)  # pc_xy B*N*2

        # img_2d = np.zeros((224, 224))
        # pc_xy_ = pc_xy.cpu().numpy()
        # for i in range(pc_xy_.shape[1]):
        #     img_2d[int(pc_xy_[0, i, 1]), int(pc_xy_[0, i, 0])] = 1
        # ipdb.set_trace()
        # img_2d = img_2d * 255
        # img_2d = img_2d.astype(np.uint8)
        # import cv2
        # cv2.imwrite('temp.png', img_2d)

        half_resolution = (self.options.map_size - 1) / 2.
        nomalized_pc_xy = ((pc_xy - half_resolution) / half_resolution).unsqueeze(1)
        outf1 = F.grid_sample(f1, nomalized_pc_xy, align_corners=True)
        outf2 = F.grid_sample(f2, nomalized_pc_xy, align_corners=True)
        outf3 = F.grid_sample(f3, nomalized_pc_xy, align_corners=True)
        outf4 = F.grid_sample(f4, nomalized_pc_xy, align_corners=True)
        outf5 = F.grid_sample(f5, nomalized_pc_xy, align_corners=True)
        # out = torch.cat((outf1, outf2, outf3, outf4, outf5), dim=1)
        return outf1, outf2, outf3, outf4, outf5

    def __repr__(self):
        return self.__class__.__name__ + ' (Map pc to ' \
               + str(self.options.map_size) + ' x ' + str(self.options.map_size) \
               + ' plane)'

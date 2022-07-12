import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.sdf_threshold = self.options.loss.sdf.threshold
        self.sdf_near_surface_weight = self.options.loss.sdf.weights.near_surface
        self.sdf_scale = self.options.loss.sdf.weights.scale

    def forward(self, pred_dict, input_batch):
        # import ipdb; ipdb.set_trace()
        sample_pc = input_batch["sdf_points"]
        regress_mat = input_batch["regress_mat"]
        pred_RT = pred_dict["pred_RT"]
        rot_mat_inv = input_batch["rot_mat_inv"]
        norm_mat_inv = input_batch["norm_mat"]
        # import ipdb;ipdb.set_trace()
        pred_regress_mat = norm_mat_inv @ rot_mat_inv @ pred_RT
        K = input_batch["K"]
        pred_trans_mat = pred_regress_mat @ K

        ones = torch.ones((sample_pc.shape[0], sample_pc.shape[1], 1), dtype=torch.float32, device=sample_pc.device)
        homo_sample_pc = torch.cat([sample_pc, ones], dim=-1)

        pred_3d = torch.matmul(homo_sample_pc, pred_regress_mat)
        gt_3d = torch.matmul(homo_sample_pc, regress_mat)

        rotpc_loss = F.mse_loss(pred_3d, gt_3d)

        loss = rotpc_loss

        return loss, {
            "loss": loss,
            "rotpc_loss": rotpc_loss,
        }

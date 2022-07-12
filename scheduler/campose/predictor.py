from scheduler.base.base_predictor import Predictor

from models.zoo.posenet import PoseNet
import os
import numpy as np
import struct
import torch
import torch.nn.functional as F


class PoseNetPredictor(Predictor):
    
    def init_model(self):
        return PoseNet(self.options.model)

    def predict_step(self, input_batch):
        self.model.eval()
        # Run inference
        with torch.no_grad():
            pred_dict = self.model(input_batch)
        
        sample_pc = input_batch["sdf_points"]
        trans_mat = input_batch['trans_mat']
        regress_mat = input_batch["regress_mat"]
        pred_RT = pred_dict["pred_RT"]
        rot_mat_inv = input_batch["rot_mat_inv"]
        norm_mat_inv = input_batch["norm_mat"]
        pred_regress_mat = norm_mat_inv @ rot_mat_inv @ pred_RT
        K = input_batch["K"]
        ones = torch.ones((sample_pc.shape[0], sample_pc.shape[1], 1), dtype=torch.float32, device=sample_pc.device)
        homo_sample_pc = torch.cat([sample_pc, ones], dim=-1)

        pred_3d = torch.matmul(homo_sample_pc, pred_regress_mat)
        gt_3d = torch.matmul(homo_sample_pc, regress_mat)

        d3D = torch.sqrt(torch.square((pred_3d - gt_3d)).sum(-1)).mean()
        pred_trans_mat = pred_regress_mat @ K
        pred_sample_img_points, pred_xy = self.get_img_points(sample_pc, pred_trans_mat)
        gt_sample_img_points, gt_xy = self.get_img_points(sample_pc, trans_mat)
        d2D = torch.sqrt(torch.square((pred_sample_img_points - gt_sample_img_points)).sum(-1)).mean()
        return d3D.detach().item(), d2D.detach().item()

    def predict(self):
        self.logger.info("Running predictions...")
        predict_data_loader = self.get_dataloader()
        L_d3D, L_d2D = [], []
        for step, batch in enumerate(predict_data_loader):
            self.logger.info("Predicting [%05d/%05d]" % (step * self.options.test.batch_size, len(self.dataset)))

            if self.gpu_inference:
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                raise NotImplementedError("CPU inference is currently buggy. This takes some extra efforts and "
                                          "might be fixed in the future.")
            d3D, d2D = self.predict_step(batch)
            L_d3D.append(d3D)
            L_d2D.append(d2D)
        
        print('d3D:', np.mean(np.array(L_d3D)))
        print('d2D:', np.mean(np.array(L_d2D)))


    def get_img_points(self, sample_pc, trans_mat_right, img_size = (137,137)):
        # sample_pc B*N*3
        size_lst = sample_pc.shape
        device = sample_pc.device
        ones =  torch.ones([size_lst[0], size_lst[1], 1], dtype=torch.float32, device=device)
        homo_pc = torch.cat([sample_pc, ones], dim=-1)
        # print("homo_pc.get_shape()", homo_pc.shape)
        pc_xyz = torch.matmul(homo_pc, trans_mat_right)
        # print("pc_xyz.get_shape()", pc_xyz.shape) # B * N * 3
        pc_xy = pc_xyz[:,:,:2] / pc_xyz[:,:,2:]
        
        pred_sample_img_points = torch.clamp(pc_xy, min=0.0, max=img_size[0])
        
        return pred_sample_img_points, pc_xy
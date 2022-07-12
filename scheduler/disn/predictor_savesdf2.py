from scheduler.base.base_predictor import Predictor

from models.zoo.disn import DISNmodel
import os
import numpy as np
import struct
import torch


class DISNPredictor(Predictor):
    def init_auxiliary(self):
        os.environ["LD_LIBRARY_PATH"] += os.pathsep + os.getcwd() + "/external/isosurface"

    def init_model(self):
        return DISNmodel(self.options.model)

    def predict_step(self, input_batch):
        self.model.eval()
        filenames = input_batch["filename"]
        if not filenames[0].endswith('_0'):
            return

        total_points = self.options.model.disn.resolution ** 3
        split_size = int(np.ceil(total_points / self.options.model.disn.split_chunk))  # 80
        num_sample_points = int(np.ceil(total_points / split_size))  # 257**3 / 80
        batch_size = input_batch["images"].shape[0]
        extra_pts = np.zeros((1, split_size * num_sample_points - total_points, 3), dtype=np.float32)
        batch_points = np.zeros((split_size, 0, num_sample_points, 3), dtype=np.float32)
        for b in range(batch_size):
            sdf_params = input_batch['sdf_params'][b].cpu()
            x_ = np.linspace(sdf_params[0], sdf_params[3], num=self.options.model.disn.resolution)
            y_ = np.linspace(sdf_params[1], sdf_params[4], num=self.options.model.disn.resolution)
            z_ = np.linspace(sdf_params[2], sdf_params[5], num=self.options.model.disn.resolution)
            z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
            x = np.expand_dims(x, 3)
            y = np.expand_dims(y, 3)
            z = np.expand_dims(z, 3)
            all_pts = np.concatenate((x, y, z), axis=3).astype(np.float32)
            all_pts = all_pts.reshape(1, -1, 3)
            all_pts = np.concatenate((all_pts, extra_pts), axis=1).reshape(split_size, 1, -1, 3)
            self.logger.info('all_pts: {}'.format(all_pts.shape))
            batch_points = np.concatenate((batch_points, all_pts), axis=1)

        pred_sdf_val_all = np.zeros((split_size, batch_size, 1, num_sample_points))

        with torch.no_grad():
            for sp in range(split_size):
                images = input_batch["images"]
                trans_mat = input_batch["trans_mat"]

                pc = torch.tensor(batch_points[sp, ...].reshape(batch_size, -1, 3), device=images.device)
                pc_rot = torch.tensor(batch_points[sp, ...].reshape(batch_size, -1, 3), device=images.device)
                input_batch["sdf_points"] = pc
                input_batch["sdf_points_rot"] = pc_rot
                out = self.model(input_batch)
                pred_sdf_val = out["pred_sdf"]
                pred_sdf_val_all[sp, :, :, :] = pred_sdf_val.cpu().numpy()
        # [split_size, batch, num_sample_points, 1]->[B,S,N,1]->[B,1,SN]
        pred_sdf_val_all = np.swapaxes(pred_sdf_val_all, 0, 1)  # B, S, C=1, NUM SAMPLE
        pred_sdf_val_all = pred_sdf_val_all.reshape((batch_size, 1, -1))[:, :, :total_points]
        result = pred_sdf_val_all / self.options.loss.sdf.weights.scale  # [B,1,total]
        res = self.options.model.disn.resolution
        result = result.reshape((batch_size, res, res, res))
        # reshape to original shape
        # iso = 0.003
        for b in range(batch_size):
            obj_nm = input_batch['filename'][b]
            sdf_params = input_batch['sdf_params'][b].cpu().numpy()
            save_path = os.path.join(self.options.predict_dir, obj_nm + ".npy")
            np.save(save_path, result[b])
            np.save(save_path.replace('.npy','_sdf_params.npy'), sdf_params)

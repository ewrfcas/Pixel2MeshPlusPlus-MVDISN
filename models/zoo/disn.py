import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import get_backbone
from models.layers import PerceptualPooling, PointMLP, SDFDecoder


class DISNmodel(nn.Module):
    def __init__(self, options):
        super(DISNmodel, self).__init__()
        self.options = options
        self.nn_encoder = get_backbone(self.options)
        self.percep_pooling = PerceptualPooling(self.options)

        self.point_mlp_global = PointMLP(self.options)
        # self.point_mlp_local = PointMLP(self.options)

        self.nn_decoder = SDFDecoder(self.options, feat_channel=512 + 1024 + 448)

    def forward(self, input_batch):
        # import ipdb; ipdb.set_trace()
        img = input_batch["images"]    # B,V,C,H,W
        pc = input_batch["sdf_points"]  # B,N,3
        pc_rot = input_batch["sdf_points_rot"] # B,N,3
        trans_mat = input_batch["trans_mat"] # B,V,4,3
        sdf_value = input_batch['sdf_value']

        batch_size = img.size(0)
        view_num = img.size(1)

        all_latent_feat = []
        for v in range(view_num):
            img_v = img[:, v] # B,C,H,W
            trans_mat_v = trans_mat[:, v] # B,4,3
            img_featuremaps, img_feat_global = self.nn_encoder(img_v)
            outf1, outf2, outf3, outf4, outf5 = self.percep_pooling(img_featuremaps, pc, trans_mat_v, sdf_value)
            img_feat_local = torch.cat((outf1, outf2, outf3), dim=1)
            # change B*N*3 --> B*3*1*N
            pc_rot_v = pc_rot.unsqueeze(3).permute(0, 2, 3, 1)
            point_feat_global = self.point_mlp_global(pc_rot_v)
            num_points = point_feat_global.shape[-1]
            expand_img_feat_global = img_feat_global.expand(-1, -1, -1, num_points)
            latent_feat_v = torch.cat([point_feat_global, expand_img_feat_global, img_feat_local], dim=1)
            all_latent_feat.append(latent_feat_v)
        # B, V, C, 1, N
        all_latent_feat = torch.stack(all_latent_feat, dim=1)
        # max pooling for all latent code
        max_latent = torch.max(all_latent_feat, dim=1)[0]
        pred_sdf = self.nn_decoder(max_latent)


        if self.options.tanh:
            pred_sdf = F.tanh(pred_sdf)
        pred_sdf = pred_sdf.squeeze(2)

        return {
            "pred_sdf": pred_sdf
        }

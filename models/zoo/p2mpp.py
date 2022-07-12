import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.backbones import get_backbone
from models.layers.gbottleneck import GBottleneck, DeformationReasoning
from models.layers.gconv import GConv
from models.layers.gpooling import GUnpooling
from models.layers.gprojection import GProjection

from utils.mesh import camera_info
from utils.tensor import reduce_std

from pytorch3d.utils import ico_sphere
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds

from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRasterizer
)

def make_mesh_transformed_to_N_views(mesh_batch, cams):
    res = []
    batch_size = mesh_batch.shape[0]
    num_views = cams.shape[1]
    cams = cams.detach().cpu().numpy()
    for i in range(batch_size):
        mesh = mesh_batch[i].detach().cpu().numpy()
        cams_ori = cams[i][0]
        R_ori, T_ori = camera_info(cams_ori)
        mesh_ori = mesh @ np.linalg.inv(R_ori.T) + T_ori
        mesh_N_views = []
        for v in range(num_views):
            R_v, T_v = camera_info(cams[i][v])
            mesh_v = np.dot(mesh_ori-T_v, R_v.T)
            mesh_N_views.append(torch.tensor(mesh_v, dtype=torch.float32, device=mesh_batch.device))
        mesh_N_views = torch.stack(mesh_N_views, 1)
        res.append(mesh_N_views)
    res = torch.stack(res, 0)
    return res

class P2MPPModel(nn.Module):

    def __init__(self, options, ellipsoid, camera_f, camera_c, mesh_pos):
        super(P2MPPModel, self).__init__()

        self.hidden_dim = options.hidden_dim
        self.coord_dim = options.coord_dim
        self.last_hidden_dim = options.last_hidden_dim
        self.init_pts = nn.Parameter(ellipsoid.coord, requires_grad=False)
        self.gconv_activation = options.gconv_activation

        self.nn_encoder = get_backbone(options)
        # there are 4 stages, all features we use [ [64, 64, 128, 256, 512], 3]

        self.features_dim = self.nn_encoder.features_dim + self.coord_dim

        self.local_gcns = nn.ModuleList([
            DeformationReasoning(3 + 2*(16+32+64), 192, 1, ellipsoid.sample_adj_mat,
                                 sample_coord=ellipsoid.sample_coord),
            DeformationReasoning(3 + 2*(16+32+64), 192, 1, ellipsoid.sample_adj_mat,
                                 sample_coord=ellipsoid.sample_coord),
            # DeformationReasoning(3 + 2*(16+32+64), 192, 1, ellipsoid.sample_adj_mat,
            #                      sample_coord=ellipsoid.sample_coord)
        ])


        self.projection = GProjection(mesh_pos, camera_f, camera_c, bound=options.z_threshold,
                                      tensorflow_compatible=options.align_with_tensorflow)

        # last gconv output final result
        # self.gconv = GConv(in_features=self.last_hidden_dim, out_features=self.coord_dim,
        #                    adj_mat=ellipsoid.adj_mat[2])
        self.ellipsoid = ellipsoid
        self.faces = self.ellipsoid.obj_fmt_faces[2][:,1:].astype(int) - 1
        raster_settings = RasterizationSettings(
            image_size=224,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.htpo_num = 42

    def forward(self, input_batch):
        """
        :param img: images BxVx3x224x224
        """
        # import ipdb; ipdb.set_trace()
        # assume view_id = v
        # perceptual_feat = [x0, x1, x2]
        img_feats_all_views = []
        batch_size, views_num = input_batch['images'].shape[0], input_batch['images'].shape[1]
        img_shape = self.projection.image_feature_shape(input_batch['images'][:, 0])
        device = input_batch['images'].device

        for v in range(views_num):
            img_v = input_batch['images'][:, v]
            img_feats_v = self.nn_encoder(img_v)
            img_feats_all_views.append(img_feats_v)

        mesh = input_batch['mesh']

        ori_mesh = Meshes(verts=input_batch['ori_mesh_verts'], faces=input_batch['ori_mesh_faces'])
        # (V, 3) where V is the total number of verts across all the meshes in the batch
        verts_packed = ori_mesh.verts_packed()
        vertex_visibility_map_packed = torch.zeros((views_num, verts_packed.shape[0]), device=device)
        p2mT = input_batch['p2mT']

        for i in range(views_num):
            R, T = look_at_view_transform(eye=p2mT[i])
            fragments = self.rasterizer(ori_mesh, cameras=FoVPerspectiveCameras(device=device,
                                        fov=2*np.rad2deg(np.arctan2(28.5, 2*32)),
                                        R=R[:, ...],
                                        T=T[:, ...]*1.75,))
            pix_to_face = fragments.pix_to_face
            # (F, 3) where F is the total number of faces across all the meshes in the batch
            packed_faces = ori_mesh.faces_packed()
            for b in range(batch_size):
                # Indices of unique visible faces
                visible_faces = pix_to_face[b,...,0].unique()[1:]   # (num_visible_faces )
                # Get Indices of unique visible verts using the vertex indices in the faces
                visible_verts_idx = packed_faces[visible_faces]    # (num_visible_faces,  3)
                unique_visible_verts_idx = torch.unique(visible_verts_idx)   # (num_visible_verts, )
                # Update visibility indicator to 1 for all visible vertices 
                vertex_visibility_map_packed[i, unique_visible_verts_idx] = 1.0

        vertex_visibility_list = torch.split(vertex_visibility_map_packed, [x.shape[0] for x in ori_mesh.verts_list()], dim=1)
        vertex_visibility_list = torch.stack(vertex_visibility_list, dim=0).permute(0, 2, 1)
        invis_list = (vertex_visibility_list.sum(-1) < 1).float()
        vis_list = (vertex_visibility_list.sum(-1) > 0).float()

        # import ipdb; ipdb.set_trace()

        # B, N, V, 3
        mesh_all_view = make_mesh_transformed_to_N_views(mesh, input_batch['camera_params'])
        # B, N, S, V, 3
        # TODO 这里并没有处理不同数量的mesh顶点
        all_verts_to_sample = self.ellipsoid.sample_coord[None, None, :, None, :].to(device) + mesh_all_view[:, :, None, :, :]

        hypothesis_feature_only_img_feat_all_views = []
        for v in range(views_num):
            hypothesis_reshape_v = all_verts_to_sample[:, :, :, v, :].view(batch_size, -1, 3)
            hypothesis_flatten_feature_v = self.projection(img_shape, img_feats_all_views[v], hypothesis_reshape_v)
            hypothesis_feature_v = hypothesis_flatten_feature_v.reshape(batch_size, -1, self.htpo_num, self.features_dim)
            hypothesis_feature_only_img_feat_v = hypothesis_feature_v[..., 3:]
            hypothesis_feature_only_img_feat_v = hypothesis_feature_only_img_feat_v * vertex_visibility_list[..., v, None, None]
            hypothesis_feature_only_img_feat_all_views.append(hypothesis_feature_only_img_feat_v)

        stacked_hypothesis_only_im_feat = torch.stack(hypothesis_feature_only_img_feat_all_views, dim=4)
        # import ipdb; ipdb.set_trace()
        # B, N, S, F, V
        max_hypothesis_only_im_feat = torch.max(stacked_hypothesis_only_im_feat, 4)[0]
        mean_hypothesis_only_im_feat = torch.sum(stacked_hypothesis_only_im_feat, 4) / vertex_visibility_list.sum(-1)[:, :, None, None].clamp(min=1)
        # torch.mean(stacked_hypothesis_only_im_feat, 4)
        # std_hypothesis_only_im_feat= reduce_std(stacked_hypothesis_only_im_feat, 4)

        coord_feat = self.ellipsoid.sample_coord[None, None, :, :].to(device) + mesh[:, :, None, :]

        fused_feature = torch.cat([coord_feat,
                                   max_hypothesis_only_im_feat,
                                   mean_hypothesis_only_im_feat], -1)

        dx1, score1, x6 = self.local_gcns[0](fused_feature)

        masked_dx1 = dx1 * vis_list[..., None]
        summary_dx1 = torch.norm(masked_dx1, dim=-1).sum() / vis_list.sum()
        update_coord1 = mesh + masked_dx1

        # # ----------- 2
        # mesh2 = update_coord1
        # mesh_all_view2 = make_mesh_transformed_to_N_views(mesh2,  input_batch['camera_params'])
        # all_verts_to_sample2 = self.ellipsoid.sample_coord[None, None, :, None, :].to(device) + mesh_all_view2[:, :, None, :, :]
        #
        # hypothesis_feature_only_img_feat_all_views2 = []
        # for v in range(views_num):
        #     hypothesis_reshape_v2 = all_verts_to_sample2[:, :, :, v, :].view(batch_size, -1, 3)
        #     hypothesis_flatten_feature_v2 = self.projection(img_shape, img_feats_all_views[v], hypothesis_reshape_v2)
        #     hypothesis_feature_v2 = hypothesis_flatten_feature_v2.reshape(batch_size, -1, self.htpo_num, self.features_dim)
        #     hypothesis_feature_only_img_feat_v2 = hypothesis_feature_v2[..., 3:]
        #     hypothesis_feature_only_img_feat_v2 = hypothesis_feature_only_img_feat_v2 * vertex_visibility_list[..., v, None, None]
        #     hypothesis_feature_only_img_feat_all_views2.append(hypothesis_feature_only_img_feat_v2)
        #
        # stacked_hypothesis_only_im_feat2 = torch.stack(hypothesis_feature_only_img_feat_all_views2, dim=4)
        # max_hypothesis_only_im_feat2 = torch.max(stacked_hypothesis_only_im_feat2, 4)[0]
        # mean_hypothesis_only_im_feat2 = torch.sum(stacked_hypothesis_only_im_feat2, 4) / vertex_visibility_list.sum(-1)[:, :, None, None].clamp(min=1)
        # # torch.mean(stacked_hypothesis_only_im_feat2, 4)
        # # std_hypothesis_only_im_feat2 = reduce_std(stacked_hypothesis_only_im_feat2, 4)
        #
        # coord_feat2 = self.ellipsoid.sample_coord[None, None, :, :].to(device) + mesh2[:, :, None, :]
        # fused_feature2 = torch.cat([coord_feat2,
        #                             max_hypothesis_only_im_feat2,
        #                             mean_hypothesis_only_im_feat2], -1)
        # dx2, score2, x62 = self.local_gcns[1](fused_feature2)
        #
        # masked_dx2 = dx2 * vis_list[..., None]
        # summary_dx2 = torch.norm(masked_dx2, dim=-1).sum() / vis_list.sum()
        # update_coord2 = mesh2 + masked_dx2
        # mesh2 = update_coord2

        pred_coord = update_coord1
        # import ipdb; ipdb.set_trace()

        return {
            "pred_coord": [pred_coord],
            "pred_coord_before_deform": [mesh],
            "reconst": None,
            "score1": score1,
            "dx1": summary_dx1,
            "x6": x6,
            # "score2": score2,
            # "dx2": summary_dx2,
            # "x62": x62,
        }

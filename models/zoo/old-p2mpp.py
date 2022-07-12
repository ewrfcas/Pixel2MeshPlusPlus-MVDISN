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
                                 sample_coord=ellipsoid.sample_coord)
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

    def forward(self, input_batch):
        """
        :param img: images BxVx3x224x224
        """
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


        # img = input_batch['images'][:, 0]
        # batch_size = img.size(0)
        # img_feats = self.nn_encoder(img)
        # img_shape = self.projection.image_feature_shape(img)

        # # extract feature for v-1
        # img_pre = input_batch['images'][:, 1]
        # img_pre_feats = self.nn_encoder(img_pre)

        # # extract feature for v+1
        # img_nxt = input_batch['images'][:, 2]
        # img_nxt_feats = self.nn_encoder(img_nxt)

        # visibility check
        # generate mesh for render
        # ori_mesh = Meshes(verts=input_batch['ori_mesh_verts'], faces=input_batch['ori_mesh_faces'])
        # (V, 3) where V is the total number of verts across all the meshes in the batch
        # verts_packed = ori_mesh.verts_packed()

        # num_views = input_batch['images'].shape[1]
        # vertex_visibility_map_packed = torch.zeros((num_views, verts_packed.shape[0]), device=img.device)
                
        # p2mT = input_batch['p2mT']
        
        # for i in range(3):
        #     R, T = look_at_view_transform(eye=p2mT[i])
        #     fragments = self.rasterizer(ori_mesh, cameras=FoVPerspectiveCameras(device=img.device,
        #                                 fov=2*np.rad2deg(np.arctan2(28.5, 2*32)),
        #                                 R=R[:, ...], 
        #                                 T=T[:, ...]*1.75,))
        #     pix_to_face = fragments.pix_to_face
        #     # (F, 3) where F is the total number of faces across all the meshes in the batch
        #     packed_faces = ori_mesh.faces_packed()
        #     for b in range(batch_size):
        #         # Indices of unique visible faces
        #         visible_faces = pix_to_face[b,...,0].unique()[1:]   # (num_visible_faces )
        #         # Get Indices of unique visible verts using the vertex indices in the faces
        #         visible_verts_idx = packed_faces[visible_faces]    # (num_visible_faces,  3)
        #         unique_visible_verts_idx = torch.unique(visible_verts_idx)   # (num_visible_verts, )
        #         # Update visibility indicator to 1 for all visible vertices 
        #         vertex_visibility_map_packed[i, unique_visible_verts_idx] = 1.0

        # vertex_visibility_list = torch.split(vertex_visibility_map_packed, [x.shape[0] for x in ori_mesh.verts_list()], dim=1)
        # vertex_visibility_list = torch.stack(vertex_visibility_list, dim=0).permute(0, 2, 1)
        mesh = input_batch['mesh']
        # invis_list = (vertex_visibility_list.sum(-1) < 1).float()
        # vis_list = (vertex_visibility_list.sum(-1) > 0).float()
        # B, N, V, 3
        mesh_all_view = make_mesh_transformed_to_N_views(mesh,  input_batch['camera_params'])
        # B, N, S, V, 3
        # B,N个点, S个hypo,V,3
        # TODO: 这里并没有处理不同数量的mesh 顶点，稍后再做
        all_verts_to_sample = self.ellipsoid.sample_coord[None, None, :, None, :].to(device) + mesh_all_view[:, :, None, :, :]
        
        # hypothesis_reshape_ori = all_verts_to_sample[:, :, :, 0, :].view(batch_size, -1, 3)
        # hypothesis_reshape_pre = all_verts_to_sample[:, :, :, 1, :].view(batch_size, -1, 3)
        # hypothesis_reshape_nxt = all_verts_to_sample[:, :, :, 1, :].view(batch_size, -1, 3) # WTF BUG!

        hypothesis_flatten_feature_ori = self.projection(img_shape, img_feats, hypothesis_reshape_ori)
        hypothesis_feature_ori = hypothesis_flatten_feature_ori.reshape(batch_size, -1, 43, self.features_dim)
        hypothesis_feature_ori_only_im_feat = hypothesis_feature_ori[..., 3:]
        
        hypothesis_flatten_feature_pre = self.projection(img_shape, img_pre_feats, hypothesis_reshape_pre)
        hypothesis_feature_pre = hypothesis_flatten_feature_pre.reshape(batch_size, -1, 43, self.features_dim)
        hypothesis_feature_pre_only_im_feat = hypothesis_feature_pre[..., 3:]
        
        hypothesis_flatten_feature_nxt = self.projection(img_shape, img_nxt_feats, hypothesis_reshape_nxt)
        hypothesis_feature_nxt = hypothesis_flatten_feature_nxt.reshape(batch_size, -1, 43, self.features_dim)
        hypothesis_feature_nxt_only_im_feat = hypothesis_feature_nxt[..., 3:]

        stacked_hypothesis_only_im_feat = torch.stack([hypothesis_feature_ori_only_im_feat,
                                                  hypothesis_feature_pre_only_im_feat,
                                                  hypothesis_feature_nxt_only_im_feat], dim=4)

        max_hypothesis_only_im_feat = torch.max(stacked_hypothesis_only_im_feat, 4)[0]
        mean_hypothesis_only_im_feat = torch.mean(stacked_hypothesis_only_im_feat, 4)
        std_hypothesis_only_im_feat= torch.std(stacked_hypothesis_only_im_feat, 4, unbiased=False)
        
        # masked_hypothesis_feature_ori = hypothesis_feature_ori #* vertex_visibility_list[..., 0, None, None]
        # masked_hypothesis_feature_pre = hypothesis_feature_pre #* vertex_visibility_list[..., 1, None, None]
        # masked_hypothesis_feature_nxt = hypothesis_feature_nxt #* vertex_visibility_list[..., 2, None, None]
        #import ipdb; ipdb.set_trace()
        #all_view_features = (masked_hypothesis_feature_ori + masked_hypothesis_feature_pre + masked_hypothesis_feature_nxt)
        coord_feat = self.ellipsoid.sample_coord[None, None, :, :].to(img.device) + mesh[:,:,None,:]
        fused_feature = torch.cat([coord_feat, 
                                   max_hypothesis_only_im_feat,
                                   mean_hypothesis_only_im_feat,
                                   std_hypothesis_only_im_feat], -1)
        #all_view_features / 3 #vertex_visibility_list.sum(-1)[:, :, None, None].clamp(min=1)
        
        # torch.einsum('ij,bkjl->bkil', aa, ttt)
        # import ipdb; ipdb.set_trace()
        dx1, score1, x6 = self.local_gcns[0](fused_feature)

        masked_dx1 = dx1 #* vis_list[..., None]
        # import ipdb; ipdb.set_trace()

        update_coord1 = mesh + masked_dx1

        return {
            "pred_coord": [update_coord1],
            "pred_coord_before_deform": [mesh],
            "reconst": None,
            "score1": score1,
            "dx1": masked_dx1,
            "x6": x6
        }



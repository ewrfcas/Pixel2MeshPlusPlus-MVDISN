import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import stats
from models.layers.chamfer_wrapper import ChamferDist


class P2MPPLoss(nn.Module):
    def __init__(self, options, ellipsoid):
        super().__init__()
        self.options = options
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.chamfer_dist = ChamferDist()
        self.ellisoid = ellipsoid

        self.laplace_idx = nn.ParameterList([
            nn.Parameter(idx, requires_grad=False) for idx in ellipsoid.laplace_idx])
        self.edges = nn.ParameterList([
            nn.Parameter(edges, requires_grad=False) for edges in ellipsoid.edges])

    def edge_regularization(self, pred, edges):
        """
        :param pred: batch_size * num_points * 3
        :param edges: num_edges * 2
        :return:
        """
        return self.l2_loss(pred[:, edges[:, 0]], pred[:, edges[:, 1]]) * pred.size(-1)

    @staticmethod
    def laplace_coord(inputs, lap_idx):
        """
        :param inputs: nodes Tensor, size (n_pts, n_features = 3)
        :param lap_idx: laplace index matrix Tensor, size (n_pts, 10)
        for each vertex, the laplace vector shows: [neighbor_index * 8, self_index, neighbor_count]

        :returns
        The laplacian coordinates of input with respect to edges as in lap_idx
        """

        indices = lap_idx[:, :-2]
        invalid_mask = indices < 0
        all_valid_indices = indices.clone()
        all_valid_indices[invalid_mask] = 0  # do this to avoid negative indices

        vertices = inputs[:, all_valid_indices]
        vertices[:, invalid_mask] = 0
        neighbor_sum = torch.sum(vertices, 2)
        neighbor_count = lap_idx[:, -1].float()
        laplace = inputs - neighbor_sum / neighbor_count[None, :, None]

        return laplace

    def laplace_regularization(self, input1, input2, block_idx):
        """
        :param input1: vertices tensor before deformation
        :param input2: vertices after the deformation
        :param block_idx: idx to select laplace index matrix tensor
        :return:

        if different than 1 then adds a move loss as in the original TF code
        """

        lap1 = self.laplace_coord(input1, self.laplace_idx[block_idx])
        lap2 = self.laplace_coord(input2, self.laplace_idx[block_idx])
        laplace_loss = self.l2_loss(lap1, lap2) * lap1.size(-1)
        move_loss = self.l2_loss(input1, input2) * input1.size(-1) if block_idx > 0 else 0
        return laplace_loss, move_loss

    def normal_loss(self, gt_normal, indices, pred_points, adj_list):
        edges = F.normalize(pred_points[:, adj_list[:, 0]] - pred_points[:, adj_list[:, 1]], dim=2)
        nearest_normals = torch.stack([t[i] for t, i in zip(gt_normal, indices.long())])
        normals = F.normalize(nearest_normals[:, adj_list[:, 0]], dim=2)
        cosine = torch.abs(torch.sum(edges * normals, 2))
        return torch.mean(cosine)

    def image_loss(self, gt_img, pred_img):
        rect_loss = F.binary_cross_entropy(pred_img, gt_img)
        return rect_loss

    def forward(self, outputs, targets):
        """
        :param outputs: outputs from P2MModel
        :param targets: targets from input
        :return: loss, loss_summary (dict)
        """
        # import ipdb;ipdb.set_trace()
        chamfer_loss, edge_loss, normal_loss, lap_loss, move_loss = 0., 0., 0., 0., 0.
        lap_const = [0.2, 1., 1., ]
        edges = targets["ori_mesh_edges"]
        gt_coord, gt_normal = targets["points"], targets["normals"]
        pred_coord, pred_coord_before_deform = outputs["pred_coord"], outputs["pred_coord_before_deform"]

        verts = pred_coord[0]
        #faces= self.ellisoid.faces[2].to(verts.device)
        faces = targets['ori_mesh_faces'][0]
        pred_pts = self.sample_surface_pts(verts, faces, num=4000)

        dist1_v, dist2_v, idx1_v, idx2_v = self.chamfer_dist(gt_coord, pred_coord[0])
        dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord, pred_pts)
        
        chamfer_loss += self.options.weights.chamfer[2] * (torch.mean(dist1_v) + self.options.weights.chamfer_opposite * torch.mean(dist2_v))
        chamfer_loss += self.options.weights.chamfer[2] * (torch.mean(dist1) + torch.mean(dist2))
        # self.edges[2]
        normal_loss += self.normal_loss(gt_normal, idx2_v, pred_coord[0], edges)
        edge_loss += self.edge_regularization(pred_coord[0], edges)

        # lap, move = self.laplace_regularization(pred_coord_before_deform[0], pred_coord[0], 2)
        # lap_loss += 0.5 * lap
        # move_loss += lap_const[2] * move


        # for i in range(3):
        #     dist1, dist2, idx1, idx2 = self.chamfer_dist(gt_coord, pred_coord[i])
        #     chamfer_loss += self.options.weights.chamfer[i] * (torch.mean(dist1) + self.options.weights.chamfer_opposite * torch.mean(dist2))
        #     normal_loss += self.normal_loss(gt_normal, idx2, pred_coord[i], self.edges[i])
        #     edge_loss += self.edge_regularization(pred_coord[i], self.edges[i])
        #     lap, move = self.laplace_regularization(pred_coord_before_deform[i], pred_coord[i], i)
        #     lap_loss += lap_const[i] * lap
        #     move_loss += lap_const[i] * move

        # self.options.weights.laplace * lap_loss + \
        # self.options.weights.move * move_loss + \
        # self.options.weights.edge * edge_loss + \
        # self.options.weights.laplace * lap_loss + \
        # import ipdb;ipdb.set_trace()
        loss = chamfer_loss + \
            self.options.weights.normal * normal_loss + \
            self.options.weights.edge * edge_loss

        loss = loss * self.options.weights.constant

        return loss, {
            "loss": loss,
            "loss_chamfer": chamfer_loss,
            # "loss_edge": edge_loss,
            "loss_laplace": lap_loss,
            # "loss_move": move_loss,
            "loss_normal": normal_loss,
        }
    
    def sample_surface_pts(self, verts, faces, num=10000): 
        # import ipdb;ipdb.set_trace()
        dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())

        # calculate area of each face 
        x1,x2,x3 = torch.split(torch.index_select(verts, 1,faces[:,0]) - torch.index_select(verts, 1,faces[:,1]), 1, dim = -1)
        y1,y2,y3 = torch.split(torch.index_select(verts, 1,faces[:,1]) - torch.index_select(verts, 1,faces[:,2]), 1, dim = -1)
        A = (x2*y3-x3*y2)**2
        B = (x3*y1 - x1*y3)**2
        C = (x1*y2 - x2*y1)**2
        Areas = torch.sqrt(A+B+C)/2
        Areas = Areas / torch.sum(Areas) # percentage of each face w.r.t. full surface area 

        # define descrete distribution w.r.t. face area ratios caluclated 
        choices = np.arange(Areas.shape[1]).reshape(1, -1, 1)
        dist = stats.rv_discrete(name='custm', values=(choices, Areas.data.cpu().numpy()))
        choices = dist.rvs(size=num) # list of faces to be sampled from 

        # from each face sample a point 
        select_faces = faces[choices] 
        xs = torch.index_select(verts, 1, select_faces[:,0])
        ys = torch.index_select(verts, 1, select_faces[:,1])
        zs = torch.index_select(verts, 1, select_faces[:,2])
        uu = torch.sqrt(dist_uni.sample((num,)))
        vv = dist_uni.sample((num,))
        points = (1- uu)*xs + (uu*(1-vv))*ys + uu*vv*zs

        return points 
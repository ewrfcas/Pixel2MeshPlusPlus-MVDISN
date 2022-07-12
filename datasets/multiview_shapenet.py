import os
import pywavefront
import trimesh
import numpy as np
import torch
import h5py
from skimage import io, transform
import utils.config as config

from utils.mesh import unit, camera_info
from datasets.base_dataset import BaseDataset

import logging

pywavefront.configure_logging(
    logging.ERROR,
    formatter=logging.Formatter('%(name)s-%(levelname)s: %(message)s')
)

rot_mat_inv = np.array([[1., 0., 0.],
                        [0., 0., 1.],
                        [0., 1., 0., ]], dtype=np.float32)
fix_mat = np.array([[1., -1., -1.],
                    [1., -1., -1.],
                    [-1., 1., 1., ]], dtype=np.float32)


# TODO: make pixel2mesh dataset
class MutliViewShapeNet(BaseDataset):
    def __init__(self, file_root, file_list_name, img_dir, mesh_pos, normalization, shapenet_options, logger):
        super(MutliViewShapeNet, self).__init__()
        self.file_root = file_root
        self.mesh_pos = mesh_pos
        self.img_dir = os.path.join(self.file_root, img_dir)
        self.gt_dir = os.path.join(self.file_root, 'gt')
        self.file_names = []
        for lst in file_list_name:
            category_name = lst.split("_")[0]
            with open(os.path.join(self.file_root, "meta", lst), "r") as fp:
                for l in fp:
                    for r in range(1):
                        self.file_names.append((category_name, l.strip(), r))
        self.normalization = normalization
        self.shapenet_options = shapenet_options
        self.logger = logger
        self.tt = "/workspace/Users/wc/shapenet/ShapeNetRendering"
        # self.mesh_dir = "/workspace/Users/wc/shapenet/pami/disn_cam_est_to_p2m_objs_simply/03001627"
        self.mesh_dir = "/home/wmlce/shapenet_processed/disn_2fold_prediction/mvdisn_2folds"
        self.cam_est = False  # True
        self.pred_RT_dir = '/workspace/Users/wc/shapenet/pami/test_objs/camest_65_0.0'

    def get_img(self, img_dir, num):
        # img_h5 = os.path.join(img_dir, "%02d_sample.h5" % num)
        # with h5py.File(img_h5, 'r') as h5_f:
        #     # the image data in h5 file is in 0-1
        #     img_ori = h5_f["img"][..., :3].astype(np.float32)
        #
        #     points = h5_f["ptnm"][:, :3].astype(np.float32)
        #     normals = h5_f["ptnm"][:, 3:].astype(np.float32)
        #     norm_params = h5_f["norm_params"][:].astype(np.float32)
        #     camera_params = h5_f["camera_params"][:].astype(np.float32)

        # TODO: ccj 0510
        img_h5 = os.path.join(img_dir, "%02d_img.h5" % num)
        sdf_h5 = os.path.join(img_dir, "sdf.h5")
        with h5py.File(img_h5, 'r') as h5_f:
            # the image data in h5 file is in 0-1
            img_ori = h5_f["img_arr"][..., :3].astype(np.float32)
            img_ori = img_ori / 255.
            camera_params = h5_f["camera_params"][:].astype(np.float32)
        with h5py.File(sdf_h5, 'r') as h5_f:
            points = h5_f["ptnm"][:, :3].astype(np.float32)
            normals = h5_f["ptnm"][:, 3:].astype(np.float32)
            norm_params = h5_f["norm_params"][:].astype(np.float32)

        points -= np.array(self.mesh_pos)
        assert points.shape[0] == normals.shape[0]
        return img_ori, points, normals, norm_params, camera_params

    def __getitem__(self, index):
        cat_id, obj, num = self.file_names[index]

        imgs = []
        pts = []
        nms = []
        camera_params_list = []
        p2mR_list = []
        p2mT_list = []
        gt_p2mR_list = []
        gt_p2mT_list = []
        for view_id in [num, (num - 1) % 24, (num + 1) % 24]:
            img_dir = os.path.join(self.img_dir, cat_id, obj)
            h5_img, pt, nm, norm_params, camera_params_ori = self.get_img(img_dir, view_id)
            # real_img_dir = os.path.join(self.tt, cat_id, obj, "rendering", "%02d.png" % view_id)
            # img = io.imread(real_img_dir)
            # if img.shape[2] > 3:  # has alpha channel
            #     img[np.where(img[:, :, 3] == 0)] = 255
            # img = transform.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
            # img = img[:, :, :3].astype(np.float32)
            # to tensor
            # img_ori = img
            ## TODO: temp codes to use h5_img ccj 0510
            img_ori = h5_img
            img_ori = torch.from_numpy(np.transpose(img_ori, (2, 0, 1)))
            img_ori_normalized = self.normalize_img(img_ori) if self.normalization else img_ori
            points = torch.from_numpy(pt)
            normals = torch.from_numpy(nm)
            norm_params = torch.from_numpy(norm_params)
            camera_params = torch.from_numpy(camera_params_ori)
            p2mR, p2mT = camera_info(camera_params_ori)

            if self.cam_est:
                with h5py.File(os.path.join(self.gt_dir, cat_id, obj, "%02d_sample.h5" % view_id)) as h5f2:
                    gt_RT = h5f2["RT"][:]
                p2m_gt_R = ((rot_mat_inv @ gt_RT[:, :3].T) * fix_mat).T
                p2m_gt_T = (gt_RT[:, 3:].T / 1.75) @ p2m_gt_R

                pred_RT = \
                np.load(os.path.join(self.pred_RT_dir, cat_id, '{}_{}_{:02d}.npy'.format(cat_id, obj, view_id)))[0].T
                p2m_pred_R = ((rot_mat_inv @ pred_RT[:, :3].T) * fix_mat).T
                p2m_pred_T = ((pred_RT[:, 3:].T / 1.75) @ p2m_pred_R)[0]

            imgs.append(img_ori_normalized)
            pts.append(points)
            nms.append(normals)
            camera_params_list.append(camera_params)
            if self.cam_est:
                p2mR_list.append(p2m_pred_R)
                p2mT_list.append(p2m_pred_T)
                gt_p2mR_list.append(p2mR)
                gt_p2mT_list.append(p2mT)
            else:
                p2mR_list.append(p2mR)
                p2mT_list.append(p2mT)
                gt_p2mR_list.append(p2mR)
                gt_p2mT_list.append(p2mT)

        filename = '_'.join(map(lambda x: str(x), [cat_id, obj, "%02d" % num]))

        # load coarse mesh
        obj_name = '{}_{}_{}.obj'.format(cat_id, obj, "%d" % num)
        obj_file = trimesh.load_mesh(os.path.join(self.mesh_dir, obj_name), process=False, maintain_order=True)
        # pywavefront.Wavefront(os.path.join(self.mesh_dir, obj_name), collect_faces=True)
        obj_vertices = obj_file.vertices
        # obj_faces = np.array(obj_file.mesh_list[0].faces, dtype=np.long)
        obj_faces = np.array(obj_file.faces, dtype=np.long)
        vertices = np.array(obj_vertices, dtype=np.float32)
        obj_edges = np.array(obj_file.edges, dtype=np.long)

        p2mR_0, p2mT_0 = gt_p2mR_list[0], gt_p2mT_list[0]
        ori_mesh_verts = (vertices @ np.linalg.inv(p2mR_0.T) + p2mT_0) / 0.57
        # 变换到 canonical view 的 mesh，用于渲染
        ori_mesh_verts = torch.FloatTensor(ori_mesh_verts)
        ori_mesh_faces = torch.LongTensor(obj_faces)
        ori_mesh_edges = torch.LongTensor(obj_edges)
        mesh_vertices = torch.FloatTensor(vertices)  # view 0 下的mesh，用于pooling

        # import ipdb; ipdb.set_trace()

        return {
            # v, v-1, v+1
            "images": torch.stack(imgs, 0),
            "images_ori": imgs[0],
            "points": pts[0],
            "normals": nms[0],
            "camera_params": torch.stack(camera_params_list, 0),
            "filename": filename,
            "mesh": mesh_vertices,
            "ori_mesh_verts": ori_mesh_verts,
            "ori_mesh_faces": ori_mesh_faces,
            "ori_mesh_edges": ori_mesh_edges,
            "p2mT": p2mT_list,
            'p2mR': p2mR_list,
        }

    def __len__(self):
        return len(self.file_names)

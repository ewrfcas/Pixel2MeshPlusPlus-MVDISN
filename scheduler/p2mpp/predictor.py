from scheduler.base.base_predictor import Predictor
import os
import numpy as np
import torch
from models.zoo.p2mpp_cam_est import P2MPPModel
from utils.mesh import Ellipsoid
# from utils.vis.renderer import MeshRenderer


class P2MPPPredictor(Predictor):
    def init_auxiliary(self):
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)

    def init_model(self):
        return P2MPPModel(self.options.model, self.ellipsoid,
                        self.options.dataset.camera_f, self.options.dataset.camera_c,
                        self.options.dataset.mesh_pos)

    def predict_step(self, input_batch):
        self.model.eval()
        # Run inference
        with torch.no_grad():
            out = self.model(input_batch)
            self.save_inference_results(input_batch, out)

    def save_inference_results(self, inputs, outputs):
        # import ipdb; ipdb.set_trace()
        batch_size = inputs["images"].size(0)
        for i in range(batch_size):
            faces = inputs["ori_mesh_faces"][i].cpu().numpy()
            file_name_wo_ext = os.path.splitext(os.path.basename(inputs["filename"][i]))[0]
            basename = os.path.join(self.options.predict_dir, file_name_wo_ext)
            print(basename)
            mesh_center = np.mean(outputs["pred_coord_before_deform"][0][i].cpu().numpy(), 0)
            # verts = [outputs["pred_coord"][k][i].cpu().numpy() for k in range(3)]
            verts_refined = outputs["pred_coord"][0][i].cpu().numpy()
            meshname_refined = basename + "_refined_%d.obj" % (2 + 1)
            vert_v_refined = np.hstack((np.full([verts_refined.shape[0], 1], "v"), verts_refined))
            # self.ellipsoid.obj_fmt_faces[2]
            # import ipdb; ipdb.set_trace()
            faces_f = np.hstack((np.full([faces.shape[0], 1], "f"), faces + 1))
            mesh_refined = np.vstack((vert_v_refined, faces_f))
            np.savetxt(meshname_refined, mesh_refined, fmt='%s', delimiter=" ")

            verts_coarse = outputs["pred_coord_before_deform"][0][i].cpu().numpy()
            meshname_coarse = basename + "_coarse_%d.obj" % (2 + 1)
            vert_v_coarse = np.hstack((np.full([verts_coarse.shape[0], 1], "v"), verts_coarse))
            mesh_coarse = np.vstack((vert_v_coarse, faces_f))
            np.savetxt(meshname_coarse, mesh_coarse, fmt='%s', delimiter=" ")
            verts_gt = inputs["points"][i].cpu().numpy()
            meshname_gt = basename + "_gt_%d.xyz" % (2 + 1)
            mesh_gt= verts_gt
            np.savetxt(meshname_gt, mesh_gt)


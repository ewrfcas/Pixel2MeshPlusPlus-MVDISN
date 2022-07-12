from scheduler.base.base_predictor import Predictor
import os
import numpy as np
import torch
from models.zoo.p2m import P2MModel
from utils.mesh import Ellipsoid
# from utils.vis.renderer import MeshRenderer


class P2MPredictor(Predictor):
    def init_auxiliary(self):
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)

    def init_model(self):
        return P2MModel(self.options.model, self.ellipsoid,
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
            file_name_wo_ext = os.path.splitext(os.path.basename(inputs["filename"][i]))[0]
            basename = os.path.join(self.options.predict_dir, file_name_wo_ext)
            print(basename)
            mesh_center = np.mean(outputs["pred_coord_before_deform"][0][i].cpu().numpy(), 0)
            verts = [outputs["pred_coord"][k][i].cpu().numpy() for k in range(3)]
            k = 2
            for vert in verts[2:]:
                meshname = basename + "_%d.obj" % (k + 1)
                vert_v = np.hstack((np.full([vert.shape[0], 1], "v"), vert))
                mesh = np.vstack((vert_v, self.ellipsoid.obj_fmt_faces[k]))
                np.savetxt(meshname, mesh, fmt='%s', delimiter=" ")


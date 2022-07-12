from scheduler.base.base_trainer import Trainer
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.nn
from models.zoo.p2mpp_cam_est import P2MPPModel
from models.losses.p2mpp_loss import P2MPPLoss
from utils.mesh import Ellipsoid


# from utils.vis.renderer import MeshRenderer


class P2MPPTrainer(Trainer):
    def init_auxiliary(self):
        # create renderer
        # self.renderer = MeshRenderer(self.options.dataset.camera_f, self.options.dataset.camera_c,
        #                              self.options.dataset.mesh_pos)
        # create ellipsoid
        self.ellipsoid = Ellipsoid(self.options.dataset.mesh_pos)

    def init_model(self):
        return P2MPPModel(self.options.model, self.ellipsoid,
                          self.options.dataset.camera_f,
                          self.options.dataset.camera_c,
                          self.options.dataset.mesh_pos)

    def init_loss_functions(self):
        return P2MPPLoss(self.options.loss, self.ellipsoid).cuda()

    def get_dataloader(self):
        if self.options.distributed:
            sampler_train = DistributedSampler(self.dataset)
        else:
            sampler_train = torch.utils.data.RandomSampler(self.dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, self.options.train.batch_size,
                                                            drop_last=True)
        data_loader = DataLoader(self.dataset, batch_sampler=batch_sampler_train,
                                 num_workers=self.options.num_workers,
                                 # num_workers=0,
                                 pin_memory=self.options.pin_memory)
        self.sampler_train = sampler_train
        return data_loader

    def train_summaries(self, input_batch, out_summary, loss_summary):
        # Debug info for filenames
        self.logger.debug(input_batch["filename"])
        # Save results in Tensorboard
        self.summary_writer.add_scalar('dx1', out_summary['dx1'], self.step_count)
        # self.summary_writer.add_scalar('dx2', out_summary['dx2'], self.step_count)
        # self.summary_writer.add_scalar('dx3', torch.mean(torch.norm(out_summary['dx3'], dim=-1)), self.step_count)
        self.summary_writer.add_histogram('score1', out_summary['score1'], self.step_count)
        # self.summary_writer.add_histogram('score2', out_summary['score2'], self.step_count)
        # self.summary_writer.add_histogram('score3', out_summary['score3'], self.step_c    ount)
        self.summary_writer.add_histogram('maxidx1', torch.max(out_summary['score1'], dim=2)[1], self.step_count)
        # self.summary_writer.add_histogram('maxidx2', torch.max(out_summary['score2'], dim=2)[1], self.step_count)
        # self.summary_writer.add_histogram('gcn_feat', out_summary['x6'], self.step_count)
        self.tensorboard_step(loss_summary)
        # Save results to log
        self.log_step(loss_summary)

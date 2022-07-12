import torch
from torch.utils.data import DataLoader, DistributedSampler

from scheduler.base.base_trainer import Trainer

from models.zoo.posenet import PoseNet
from models.losses.pose_loss import PoseLoss


class PoseNetTrainer(Trainer):
    def init_model(self):
        return PoseNet(self.options.model)

    def init_loss_functions(self):
        return PoseLoss(self.options).cuda()

    def get_dataloader(self):
        if self.options.distributed:
            sampler_train = DistributedSampler(self.dataset)
        else:
            sampler_train = torch.utils.data.RandomSampler(self.dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, self.options.train.batch_size, drop_last=True)
        data_loader = DataLoader(self.dataset, batch_sampler=batch_sampler_train,
                                 num_workers=self.options.num_workers,
                                 pin_memory=self.options.pin_memory)
        self.sampler_train = sampler_train
        return data_loader

    def log_step(self, loss_summary):
        self.logger.info("Epoch %03d/%03d, Step %06d/%06d | %06d/%06d, Time elapsed %s, Loss %.5f (AvgLoss %.5f)" % (
            self.epoch_count, self.options.train.num_epochs,
            self.step_count - ((self.epoch_count - 1) * self.dataset_size), self.dataset_size,
            self.step_count, self.options.train.num_epochs * self.dataset_size,
            self.time_elapsed, self.losses.val, self.losses.avg))

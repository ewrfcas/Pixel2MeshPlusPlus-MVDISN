import torch
from torch.utils.data import DataLoader, DistributedSampler

from scheduler.base.base_trainer import Trainer

from models.zoo.disn import DISNmodel
from models.losses.sdf_loss import SDFLoss


class DISNTrainer(Trainer):

    def init_model(self):
        return DISNmodel(self.options.model)

    def init_loss_functions(self):
        return SDFLoss(self.options).cuda()

    def get_dataloader(self, limit=None):
        if limit is not None:
            self.dataset.file_names = self.dataset.file_names[:limit]
        if self.options.distributed:
            self.sampler_train = DistributedSampler(self.dataset,
                                                    num_replicas=self.options.gpu,
                                                    rank=self.options.rank, shuffle=True)
        else:
            self.sampler_train = torch.utils.data.RandomSampler(self.dataset)
        # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, self.options.train.batch_size, drop_last=True)
        data_loader = DataLoader(self.dataset,  # batch_sampler=batch_sampler_train,
                                 batch_size=self.options.train.batch_size,
                                 sampler=self.sampler_train,
                                 drop_last=True,
                                 num_workers=self.options.num_workers,
                                 pin_memory=self.options.pin_memory)
        return data_loader

    def log_step(self, loss_summary):
        self.logger.info(
            "Epoch %03d/%03d, Step %06d/%06d | %06d/%06d, Time elapsed %s, Loss %.5f (AvgLoss %.5f), Realvalue %.5f, Acc %.3f" % (
                self.epoch_count, self.options.train.num_epochs,
                self.step_count - ((self.epoch_count - 1) * self.dataset_size), self.dataset_size,
                self.step_count, self.options.train.num_epochs * self.dataset_size,
                self.time_elapsed, self.losses.val, self.losses.avg,
                loss_summary["sdf_loss_realvalue"], loss_summary["accuracy"] * 100))

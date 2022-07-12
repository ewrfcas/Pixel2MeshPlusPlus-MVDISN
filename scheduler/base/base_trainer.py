from utils.tensor import recursive_detach
from utils.average_meter import AverageMeter
from scheduler.base.runner import CheckpointRunner
from collections import defaultdict
import numpy as np
import os
import torch
from tqdm import tqdm

try:
    from apex import amp
    from apex.parallel import convert_syncbn_model
    from apex.parallel import DistributedDataParallel

    amp.register_float_function(torch, 'matmul')
except:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from utils.misc import is_main_process


class Trainer(CheckpointRunner):
    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        # Create auxiliary models
        self.init_auxiliary()
        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = self.init_model().to('cuda')
            self.model_without_ddp = self.model
            if self.options.distributed:
                if not self.options.float16:
                    self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                           device_ids=[self.options.rank],
                                                                           output_device=self.options.rank)
                else:
                    self.model = convert_syncbn_model(self.model)
                    self.model = DistributedDataParallel(self.model)  # delay_allreduce=True
                self.model_without_ddp = self.model.module
        # Setup a joint optimizer for the 2 models
        self.optimizer = self.init_optimizer(self.options.optim.name)
        if self.options.float16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
        self.lr_scheduler = self.init_lr(self.options.optim.lr_scheduler)
        # Create loss functions
        self.criterion = self.init_loss_functions()
        # self.criterion = DataParallelCriterion(self.criterion.cuda(), device_ids=self.gpus)
        # Create AverageMeters for losses
        self.losses = AverageMeter()
        # Evaluators
        # self.evaluators = [Evaluator(self.options, self.logger, self.summary_writer, shared_model=self.model)]
        self.dataset_size = None

    def init_auxiliary(self):
        pass

    def init_model(self):
        raise NotImplementedError("Your model is not found")

    def init_loss_functions(self):
        raise NotImplementedError("Your loss is not found")

    def init_optimizer(self, optim_name):
        if optim_name == "adam":
            optimizer = torch.optim.Adam(
                # params=list([p for n, p in self.model.named_parameters() if "nn_encoder" not in n and p.requires_grad]),
                params=list(self.model_without_ddp.parameters()),
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        elif optim_name == 'adamw':
            from .AdamW import AdamW
            no_decay = ['bias']
            param_optimizer = self.model_without_ddp.named_parameters()
            optimizer_parameters = [
                {'params': [p for n, p in param_optimizer if not any([nd in n for nd in no_decay])],
                 'weight_decay': self.options.optim.wd},
                {'params': [p for n, p in param_optimizer if any([nd in n for nd in no_decay])],
                 'weight_decay': 0.0}
            ]
            optimizer = AdamW(
                params=optimizer_parameters,
                lr=self.options.optim.lr,
                betas=(self.options.optim.adam_beta1, 0.999),
                weight_decay=self.options.optim.wd
            )
        else:
            raise NotImplementedError("Your optimizer is not found")
        return optimizer

    def init_lr(self, lr_scheduler_name):
        if lr_scheduler_name == "multistep":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.options.optim.lr_step, self.options.optim.lr_factor)
        elif lr_scheduler_name == "exp":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.options.optim.lr_gamma)
        else:
            lr_scheduler = None

        return lr_scheduler

    def models_dict(self):
        return {'model': self.model_without_ddp}

    def optimizers_dict(self):
        return {'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler}

    def train_step(self, input_batch):
        # Grab data from the batch, predict with model
        out = self.model(input_batch)
        # compute loss
        if self.options.float16:
            for k in out:
                out[k] = out[k].to(torch.float32)
        loss, loss_summary = self.criterion(out, input_batch)
        self.losses.update(loss.detach().cpu().item())
        # Do backprop
        self.optimizer.zero_grad()
        if self.options.float16:
            with amp.scale_loss(loss, self.optimizer, loss_id=0) as loss_scaled:
                loss_scaled.backward()
        else:
            loss.backward()
        self.optimizer.step()
        # Pack output arguments to be used for visualization
        return recursive_detach(out), recursive_detach(loss_summary)

    def get_dataloader(self):
        raise NotImplementedError

    def train(self):
        if is_main_process():
            self.logger.info("Start Trainning.")
        # Create data loader at very begining
        train_data_loader = self.get_dataloader()
        self.dataset_size = len(train_data_loader)

        # Run training for num_epochs epochs
        for epoch in range(self.epoch_count, self.options.train.num_epochs):
            self.epoch_count += 1
            # Reset loss
            if self.options.distributed:
                self.sampler_train.set_epoch(epoch)
            self.losses.reset()
            # Iterate over all batches in an epoch
            for step, batch in enumerate(train_data_loader):
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # Run training step
                out = self.train_step(batch)
                self.step_count += 1
                # import ipdb; ipdb.set_trace()
                # Tensorboard logging every summary_steps steps
                if is_main_process() and self.step_count % self.options.train.summary_steps == 0:
                    self.train_summaries(batch, *out)
                # Save checkpoint every checkpoint_steps steps
                if is_main_process() and self.step_count % self.options.train.checkpoint_steps == 0:
                    self.dump_checkpoint(only_last=True)

            # save checkpoint after each epoch
            if is_main_process():
                self.dump_checkpoint(only_last=True)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def train_summaries(self, input_batch, out_summary, loss_summary):
        # Debug info for filenames
        self.logger.debug(input_batch["filename"])
        # Save results in Tensorboard
        # self.summary_writer.add_scalar('dx1', out_summary['dx1'], self.step_count)
        # self.summary_writer.add_scalar('dx2', out_summary['dx2'], self.step_count)
        # # self.summary_writer.add_scalar('dx3', torch.mean(torch.norm(out_summary['dx3'], dim=-1)), self.step_count)
        # self.summary_writer.add_histogram('score1', out_summary['score1'], self.step_count)
        # self.summary_writer.add_histogram('score2', out_summary['score2'], self.step_count)
        # # self.summary_writer.add_histogram('score3', out_summary['score3'], self.step_c    ount)
        # self.summary_writer.add_histogram('maxidx1', torch.max(out_summary['score1'], dim=2)[1], self.step_count)
        # self.summary_writer.add_histogram('maxidx2', torch.max(out_summary['score2'], dim=2)[1], self.step_count)
        # self.summary_writer.add_histogram('gcn_feat', out_summary['x6'], self.step_count)
        self.tensorboard_step(loss_summary)
        # Save results to log
        self.log_step(loss_summary)

    def log_step(self, loss_summary):
        self.logger.info("Epoch %03d, Step %06d/%06d, Time elapsed %s, Loss %.5f (AvgLoss %.5f)" % (
            self.epoch_count, self.step_count,
            self.options.train.num_epochs * len(self.dataset) // (self.options.train.batch_size),
            # self.options.train.num_epochs * len(self.dataset) // (
            #         self.options.train.batch_size * self.options.num_gpus),
            self.time_elapsed, self.losses.val, self.losses.avg))

    def val_step(self, input_batch):
        self.model.eval()
        out = self.model(input_batch)
        # compute loss
        if self.options.float16:
            for k in out:
                out[k] = out[k].to(torch.float32)
        _, loss_summary = self.criterion(out, input_batch)
        res = {k: v.detach().item() for k, v in loss_summary.items()}
        return res

    def validate(self, limit):
        # Create data loader at very begining
        train_data_loader = self.get_dataloader(limit=limit)
        self.dataset_size = len(train_data_loader)

        # Iterate over all batches in an epoch
        loss_dict = defaultdict(list)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(train_data_loader)):
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                # Run validate step
                out = self.val_step(batch)
                for k in out:
                    loss_dict[k].append(out[k])

        print('Validating Results...')
        for k in loss_dict:
            print(k, ':', np.mean(loss_dict[k]))

    def tensorboard_step(self, loss_summary):
        for k, v in loss_summary.items():
            self.summary_writer.add_scalar(k, v, self.step_count)

    def init_with_pretrained_backbone(self):
        checkpoint_file = os.path.abspath(self.options.train.backbone_pretrained_model)
        pretrained_dict = torch.load(checkpoint_file)
        self.model.module.load_state_dict(pretrained_dict, strict=False)
        self.logger.info("Init with pre-trained backbone from %s." % checkpoint_file)

    def test(self):
        self.model.eval()
        for evaluator in self.evaluators:
            evaluator.evaluate()
        self.model.train()

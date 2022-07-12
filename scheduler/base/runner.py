import os
from re import M
import time
from datetime import timedelta
from logging import Logger
import datetime
import torch
import torch.nn
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import default_collate

import utils.config as config
from datasets.p2m_shapenet import ShapeNet
from datasets.sdf_shapenet import ShapeNet as SDFShapeNet
from datasets.threedgan_shapenet import ThreeDGANShapeNet
from datasets.shapenet_image_folder import ShapeNetImageFolder
from datasets.multiview_shapenet import MutliViewShapeNet
from datasets.pose_shapenet import PoseShapeNet
from datasets.mv_disn_shapenet import MVDISNShapeNet
from datasets.mv_disn_ABC import MVDISNABC
from datasets.mv_disn_colmap import MvdisnColmap

from scheduler.base.saver import CheckpointSaver
from utils.misc import is_main_process


class CheckpointRunner(object):
    def __init__(self, options, logger: Logger, summary_writer: SummaryWriter, dataset=None,
                 training=True, shared_model=None):
        self.options = options
        self.logger = logger

        # GPUs
        if not torch.cuda.is_available() and self.options.num_gpus > 0:
            raise ValueError("CUDA not found yet number of GPUs is set to be greater than 0")
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            if is_main_process():
                logger.info("CUDA visible devices is activated here, number of GPU setting is not working")
            self.gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            self.options.num_gpus = len(self.gpus)
            enumerate_gpus = list(range(self.options.num_gpus))
            if is_main_process():
                logger.info("CUDA is asking for " + str(self.gpus) + ", PyTorch to doing a mapping, changing it to " +
                            str(enumerate_gpus))
            self.gpus = enumerate_gpus
        else:
            self.gpus = list(range(self.options.num_gpus))
            if is_main_process():
                logger.info("Using GPUs: " + str(self.gpus))

        # initialize summary writer
        self.summary_writer = summary_writer

        # initialize dataset
        if dataset is None:
            dataset = options.dataset  # useful during training
        self.dataset = self.load_dataset(dataset, training)
        self.dataset_collate_fn = self.load_collate_fn(dataset, training)

        # by default, epoch_count = step_count = 0
        self.epoch_count = self.step_count = 0
        self.time_start = time.time()

        # override this function to define your model, optimizers etc.
        # in case you want to use a model that is defined in a trainer or other place in the code,
        # shared_model should help. in this case, checkpoint is not used
        if is_main_process():
            self.logger.info("Running model initialization...")
        self.init_fn(shared_model=shared_model)

        if shared_model is None:  # and is_main_process()
            # checkpoint is loaded if any
            self.saver = CheckpointSaver(self.logger, checkpoint_dir=str(self.options.checkpoint_dir),
                                         checkpoint_file=self.options.checkpoint)
            if self.options.train.backbone_pretrained_model is not None:
                self.init_with_pretrained_backbone()
            else:
                self.init_with_checkpoint()

    def load_dataset(self, dataset, training):
        if is_main_process():
            self.logger.info("Loading datasets: %s" % dataset.name)
        if dataset.name == "shapenet":
            return ShapeNet(config.SHAPENET_ROOT, dataset.filelist_train if training else dataset.filelist_test,
                            dataset.img_dir, dataset.mesh_pos,
                            dataset.normalization, dataset.shapenet,
                            self.logger)
        elif dataset.name == "mv_shapenet":
            return MutliViewShapeNet(config.SHAPENET_ROOT,
                                     dataset.filelist_train if training else dataset.filelist_test,
                                     dataset.img_dir, dataset.mesh_pos,
                                     dataset.normalization, dataset.shapenet,
                                     self.logger)
        elif dataset.name == "sdfshapenet":
            return SDFShapeNet(config.SHAPENET_ROOT, dataset.filelist_train if training else dataset.filelist_test,
                               dataset.img_dir, dataset.sdf_dir,
                               dataset.normalization, dataset.shapenet,
                               self.logger)
        elif dataset.name == "pose_shapenet":
            return PoseShapeNet(config.SHAPENET_ROOT, dataset.filelist_train if training else dataset.filelist_test,
                                dataset.img_dir, dataset.sdf_dir,
                                dataset.normalization, dataset.shapenet,
                                self.logger)
        elif dataset.name == "mv_disn_shapenet":
            return MVDISNShapeNet(config.SHAPENET_ROOT, dataset.filelist_train if training else dataset.filelist_test,
                                  dataset.img_dir, dataset.sdf_dir,
                                  dataset.normalization, dataset.shapenet,
                                  self.logger)
        elif dataset.name == 'mv_disn_ABC':
            return MVDISNABC(config.SHAPENET_ROOT, dataset.filelist_train if training else dataset.filelist_test,
                             dataset.img_dir, dataset.sdf_dir,
                             dataset.normalization, dataset.shapenet,
                             self.logger)
        elif dataset.name == 'mv_disn_colmap':
            return MvdisnColmap(config.SHAPENET_ROOT, dataset.filelist_train if training else dataset.filelist_test,
                                dataset.img_dir, dataset.sdf_dir,
                                dataset.normalization, dataset.shapenet,
                                self.logger)
        elif dataset.name == 'mv_disn_shapenet_test':
            return MVDISNShapeNet(config.SHAPENET_ROOT, dataset.filelist_train if training else dataset.filelist_test,
                                 dataset.img_dir, dataset.sdf_dir,
                                 dataset.normalization, dataset.shapenet,
                                 self.logger, training=False)
        elif dataset.name == "shapenet_demo":
            return ShapeNetImageFolder(dataset.predict.folder, dataset.normalization, dataset.shapenet)
        else:
            raise NotImplementedError("Unsupported dataset")

    def load_collate_fn(self, dataset, training):
        return default_collate

    def init_fn(self, shared_model=None, **kwargs):
        raise NotImplementedError('You need to provide an _init_fn method')

    # Pack models and optimizers in a dict - necessary for checkpointing
    def models_dict(self):
        return None

    def optimizers_dict(self):
        # NOTE: optimizers and models cannot have conflicting names
        return None

    def init_with_pretrained_backbone(self):
        raise NotImplementedError('You need to provide an init_with_pretrained_backbone method')

    def init_with_checkpoint(self):
        checkpoint = self.saver.load_checkpoint(self.options.rank)
        if checkpoint is None:
            if self.options.model.backbone_pretrained:
                self.logger.info("Checkpoint load from Pytorch")
            else:
                self.logger.info("Checkpoint not loaded")
            return
        method_name = self.options.model.name
        for model_name, model in self.models_dict().items():
            if model_name in checkpoint:
                if method_name == 'p2mpp' and os.path.basename(
                        self.options.checkpoint) == 'tensorflow.pth.tar':  # TODO: tmp fix
                    self.logger.info("Checkpoint load from TF baseline")
                    pretrained_dict = {k: v for k, v in checkpoint[model_name].items() if k.startswith('nn_encoder')}
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.state_dict().update(pretrained_dict)
                        model.module.load_state_dict(pretrained_dict, strict=False)
                    else:
                        model.state_dict().update(pretrained_dict)
                        model.load_state_dict(pretrained_dict, strict=False)
                else:
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.load_state_dict(checkpoint[model_name], strict=True)
                    else:
                        model.load_state_dict(checkpoint[model_name], strict=True)
        if self.optimizers_dict() is not None:
            for optimizer_name, optimizer in self.optimizers_dict().items():
                if optimizer_name in checkpoint:
                    optimizer.load_state_dict(checkpoint[optimizer_name])
        else:
            self.logger.warning("Optimizers not found in the runner, skipping...")
        if "epoch" in checkpoint:
            self.epoch_count = checkpoint["epoch"]
        if "total_step_count" in checkpoint:
            self.step_count = checkpoint["total_step_count"]

    def dump_checkpoint(self, only_last=False):
        checkpoint = {
            "epoch": self.epoch_count,
            "total_step_count": self.step_count
        }
        for model_name, model in self.models_dict().items():
            if isinstance(model, torch.nn.DataParallel):
                checkpoint[model_name] = model.module.state_dict()
            else:
                checkpoint[model_name] = model.state_dict()
            for k, v in list(checkpoint[model_name].items()):
                if isinstance(v, torch.Tensor) and v.is_sparse:
                    checkpoint[model_name].pop(k)
        if self.optimizers_dict() is not None:
            for optimizer_name, optimizer in self.optimizers_dict().items():
                checkpoint[optimizer_name] = optimizer.state_dict()
        if only_last:
            self.saver.save_checkpoint(checkpoint,
                                       "last_{}-{}-{}".format(datetime.datetime.now().year,
                                                              datetime.datetime.now().month,
                                                              datetime.datetime.now().day))
        else:
            self.saver.save_checkpoint(checkpoint, "%06d_%06d" % (self.step_count, self.epoch_count))

    @property
    def time_elapsed(self):
        return timedelta(seconds=time.time() - self.time_start)

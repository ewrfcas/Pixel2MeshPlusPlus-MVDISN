from logging import Logger
import torch
from torch.utils.data import DataLoader
from scheduler.base.runner import CheckpointRunner
from utils.parallel import DataParallelModel

try:
    from apex import amp

    amp.register_float_function(torch, 'matmul')
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class Predictor(CheckpointRunner):
    def __init__(self, options, logger: Logger, writer, shared_model=None):
        super().__init__(options, logger, writer, training=False, shared_model=shared_model)

    # noinspection PyAttributeOutsideInit
    def init_fn(self, shared_model=None, **kwargs):
        self.gpu_inference = self.options.num_gpus > 0
        if self.gpu_inference == 0:
            raise NotImplementedError("CPU inference is currently buggy. This takes some extra efforts and "
                                      "might be fixed in the future.")
        if shared_model is not None:
            self.model = shared_model
        else:
            self.init_auxiliary()
            self.model = self.init_model()
            self.model = self.model.cuda()
            if self.options.float16:
                self.model = amp.initialize(self.model, None, opt_level='O1')
            self.model = DataParallelModel(self.model, device_ids=self.gpus)

    def models_dict(self):
        return {'model': self.model}

    def init_auxiliary(self):
        pass

    def init_model(self):
        raise NotImplementedError("Your model is not found")

    def get_dataloader(self):
        data_loader = DataLoader(self.dataset,
                                 batch_size=self.options.test.batch_size,
                                 pin_memory=self.options.pin_memory,
                                 collate_fn=self.dataset_collate_fn,
                                 shuffle=False)
        return data_loader

    def predict(self):
        self.logger.info("Running predictions...")
        predict_data_loader = self.get_dataloader()
        from tqdm import tqdm
        for step, batch in enumerate(tqdm(predict_data_loader)):
            # self.logger.info("Predicting [%05d/%05d]" % (step * self.options.test.batch_size, len(self.dataset)))

            if self.gpu_inference:
                # Send input to GPU
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                raise NotImplementedError("CPU inference is currently buggy. This takes some extra efforts and "
                                          "might be fixed in the future.")
            self.predict_step(batch)

    def predict_step(self, input_batch):
        raise NotImplementedError("Your predict step function not found.")

    def save_inference_results(self, inputs, outputs):
        raise NotImplementedError("Your result saving function not found.")

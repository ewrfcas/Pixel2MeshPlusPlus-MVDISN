import argparse
import sys
import os
import torch

from utils.config import options, update_options, reset_options
from scheduler import get_trainer
from utils.logger import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Training Entrypoint')
    parser.add_argument('--options', help='experiment options file name', required=False, type=str)

    args, rest = parser.parse_known_args()
    if args.options is None:
        print("Running without options file...", file=sys.stderr)
    else:
        update_options(args.options)

    # training
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--num-epochs', help='number of epochs', type=int)
    parser.add_argument('--version', help='version of task (timestamp by default)', type=str)
    parser.add_argument('--name', help='model name', type=str)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()

    return args


def init_distributed_mode(args):
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        args.distributed = True
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12341'
        from torch.distributed import init_process_group
        init_process_group(backend="nccl")
        args.rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.rank)
        args.world_size = n_gpu
        args.gpu = n_gpu
    else:
        print('Not using distributed mode')
        args.distributed = False
        return


def main():
    args = parse_args()
    init_distributed_mode(args)
    logger, writer = reset_options(options, args)
    set_random_seed(options.seed)
    trainer = get_trainer(options, logger, writer)
    trainer.train()


if __name__ == "__main__":
    main()

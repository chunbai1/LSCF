import argparse
from utils.utils import initialize
import torch.backends.cudnn as cudnn
from dataset.ris import create_dataset
from engine.train import train_ris
from builder import train_model_builder, resume_training


cudnn.enabled = True
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--resume', type=str, default=None)


def main():
    # initialize
    args = parser.parse_args()
    cfg, rank, logger, writer = initialize(args)

    # create training and validation dataset
    trainloader, valloader, trainsampler, valset = create_dataset(cfg, rank, logger)

    # create model, optimizer, lr_scheduler, criterion
    model, optimizer, lr_scheduler, criterion = train_model_builder(cfg, logger, len(trainloader))

    previous_best = 0.0
    epoch = -1

    # resume
    if args.resume is not None:
        model, optimizer, lr_scheduler, epoch, previous_best = \
            resume_training(args.resume, model, optimizer, lr_scheduler, logger)

    train_ris(args, cfg, model, optimizer, criterion, trainloader,
                  valloader, trainsampler, rank, logger, writer,
                  previous_best, lr_scheduler, epoch)


if __name__ == '__main__':
    main()

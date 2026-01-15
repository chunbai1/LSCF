from .train import *
from .eval import *
from .predict import *


def train_engine_builder(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler, valset,
                         rank, logger, writer, previous_best, lr_scheduler, epoch):

    train_ris(args, cfg, model, optimizer, criterion, trainloader,
                  valloader, trainsampler, rank, logger, writer,
                  previous_best, lr_scheduler, epoch)


def eval_engine_builder(model, loader, tta=False):

    metrics, metrics_name = eval_ris(model, loader, tta)

    return metrics, metrics_name


def predict_engine_builder(model, loader, args):

    predict_ris(model, loader, args)
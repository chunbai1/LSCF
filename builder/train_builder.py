import os
import torch
from utils.utils import count_params
import operator
from functools import reduce
from utils.utils import init_dec_weight, load_from
from .loss import loss_builder
from .optimizer import optimizer_builder
from .scheduler import lr_scheduler_builder
from .ris import ris_model_builder



def train_model_builder(cfg, logger, iters_per_epoch):

    model = ris_model_builder(cfg)

    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == 0:
        logger.info('Total params of visual backbone: {:.2f}M'.format(count_params(model.backbone)))
        logger.info('Total params of language backbone: {:.2f}M'.format(count_params(model.text_encoder)))
        # logger.info('Total params of visual-language fusion module: {:.2f}M'.format(count_params(model.neck)))
        logger.info('Total params of decoder: {:.2f}M'.format(count_params(model.decoder)))

        logger.info('Total params: {:.2f}M\n'.format(count_params(model)))
        logger.info(model)

    # load checkpoint of decoder head
    if cfg['model']['decoder'].get('pretrained', False):
        init_dec_weight(cfg, local_rank, logger, model.decoder)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      broadcast_buffers=False,
                                                      output_device=local_rank,
                                                      find_unused_parameters=cfg['find_unused_parameters'])

    if cfg['model'].get('load_from', False):
        load_from(cfg, local_rank, logger, model)

    # build specific parameter list to optimize
    single_model = model.module
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {"params": [p for p in single_model.decoder.parameters() if p.requires_grad]},
        # # the following are the parameters of bert
        {"params": reduce(operator.concat,
                            [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                            if p.requires_grad] for i in range(10)])}
    ]


    optimizer = optimizer_builder(model, cfg['optimizer'], params_to_optimize)
    lr_scheduler = lr_scheduler_builder(optimizer, cfg['lr_scheduler'], iters_per_epoch, cfg['epochs'])
    criterion = loss_builder(cfg['criterion']).cuda(local_rank)

    model._set_static_graph()

    return model, optimizer, lr_scheduler, criterion
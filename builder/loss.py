import torch.nn as nn
import importlib
import torch

def loss_builder(criterion_cfg):
    if criterion_cfg['type'] == 'CELoss':
        if criterion_cfg['kwargs'].get('weight', False):
            criterion_cfg['kwargs']['weight'] = torch.FloatTensor(criterion_cfg['kwargs']['weight'])
        return nn.CrossEntropyLoss(**criterion_cfg['kwargs'])
    else:
        criterion = getattr(importlib.import_module('losses'), criterion_cfg['type'])
        return criterion(**criterion_cfg['kwargs'] if criterion_cfg['kwargs'] else {})
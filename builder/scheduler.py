import math
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR


def lr_scheduler_builder(optimizer, lr_scheduler_cfg, len_loader, epochs):
    type, kwargs = lr_scheduler_cfg['type'], lr_scheduler_cfg['kwargs']

    total_iters = len_loader * epochs
    update_type = kwargs.get('update_type', 'iter')
    assert update_type in ['iter', 'epoch']

    if type == 'poly':
        lr_func = lambda x: (1 - x / total_iters if update_type == 'iter' else epochs) ** kwargs.get('lr_power', 0.9)
        lr_scheduler = LambdaLR(optimizer, lr_func)

    elif type == 'cosine':
        lr_func = lambda x: 0.5 * (1. + math.cos(math.pi * x / total_iters if update_type == 'iter' else epochs))
        lr_scheduler = LambdaLR(optimizer, lr_func)

    elif type == 'step':
        assert isinstance(kwargs.get('milestones'), int)
        lr_scheduler = StepLR(optimizer, kwargs.get('milestones', int(epochs * 2 / 3)))

    elif type == 'multistep':
        assert isinstance(kwargs.get('milestones'), list)
        default_dict = {12: [8, 11],
                        24: [16, 22],
                        36: [27, 33],
                        50: [36, 45],
                        100: [72, 90]}
        lr_scheduler = MultiStepLR(optimizer, kwargs.get('milestones', default_dict[epochs]))

    else:
        raise NotImplementedError('%s lr_scheduler is not implemented' % type)

    return lr_scheduler
from torch.optim import SGD, AdamW, Adam


def optimizer_builder(model, optimizer_cfg, optim_param_list=None):
    type, kwargs = optimizer_cfg['type'], optimizer_cfg['kwargs']
    
    default_param_list = [{'params': [p for n, p in model.named_parameters() if 'decoder' not in n],
                           'lr': kwargs.get('lr', 6e-5)},
                          {'params': [p for n, p in model.named_parameters() if 'decoder' in n],
                           'lr': kwargs.get('lr', 6e-5) * kwargs.get('lr_multi', 1.0)}]

    if type == 'SGD':
        optimizer = SGD(optim_param_list if optim_param_list else default_param_list,
                        lr=kwargs.get('lr', 0.001), momentum=0.9,
                        weight_decay=kwargs.get('weight_decay', 1e-4))
    elif type == 'AdamW':
        optimizer = AdamW(optim_param_list if optim_param_list else default_param_list,
                          lr=kwargs.get('lr', 6e-5),
                          weight_decay=kwargs.get('weight_decay', 0.01))
    elif type == 'Adam':
        optimizer = Adam(optim_param_list if optim_param_list else default_param_list,
                         lr=kwargs.get('lr', 6e-5), weight_decay=kwargs.get('weight_decay', 0.01))
    else:
        raise NotImplementedError('%s optimizer is not implemented' % type)

    return optimizer
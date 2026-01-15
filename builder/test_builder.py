import os
import torch
from utils.utils import count_params
from .ris import ris_model_builder


def test_model_builder(args):
    dataset_name, backbone, neck, decoder, _, img_size = os.path.basename(args.weight_path).split('.')[:6]
    training_size = int(img_size[3:])

    text_cfg = {'type': 'bert-base-uncased'}

    # build config
    cfg = {'model': {'backbone': {'type': backbone,
                                  'pretrained': None,
                                  'kwargs': {'in_channels': 3,
                                             'window_size': 12,
                                             'num_heads_fusion': [8, 8, 8, 8],
                                             'vlf_ris': 'MSCFNet'}},
                     'text_encoder': text_cfg
                     },
           'dataset': dataset_name,
           'crop_size': training_size,
           'criterion': {'kwargs': {}}}
    if neck != 'None':
        cfg['model']['neck'] = {'type': neck}
    if decoder != 'None':
        cfg['model']['decoder'] = {'type': decoder,
                                   'kwargs': {'trans_enc': False}}


    model = ris_model_builder(cfg)

    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == 0:
        print('Loading from {}\n'.format(args.weight_path))
        print('Total params: {:.2f}M\n'.format(count_params(model)))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      broadcast_buffers=False,
                                                      output_device=local_rank)

    checkpoint = torch.load(args.weight_path)
    model.load_state_dict(checkpoint['model'])

    return model, dataset_name, training_size
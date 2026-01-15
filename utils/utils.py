import numpy as np
import logging, json
import os, yaml, pprint
import torch, random
from prettytable import PrettyTable
from .dist_helper import setup_distributed
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple
from torch.distributed import ProcessGroup
from torch import distributed as torch_dist


def set_random_seed(seed):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

logs = set()

def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))

    logger = logging.getLogger(name)
    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0

    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def initialize(args):
    set_random_seed(args.seed)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)

    logging.getLogger('torch.distributed.distributed_c10d').setLevel(logging.CRITICAL)
    logging.getLogger('pytorch_pretrained_bert.tokenization').setLevel(logging.CRITICAL)

    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)
    writer = None
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        cfg['ngpus'] = world_size
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
        os.makedirs(args.save_path, exist_ok=True)

    return cfg, rank, logger, writer


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def count_training_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def f_score(precision, recall):
    score = 2 * (precision * recall) / (precision + recall + 1e-10)
    return score

def init_dec_weight(cfg, local_rank, logger, decoder):
    if local_rank == 0:
        logger.info('Load checkpoint of decoder from {}\n'.format(cfg['model']['decoder'].get('pretrained')))

    ckpt = torch.load(cfg['model']['decoder'].get('pretrained'))['model']
    decoder.load_state_dict(ckpt)


def load_from(cfg, local_rank, logger, model):
    if local_rank == 0:
        logger.info('Load checkpoint of model from {}\n'.format(cfg['model']['load_from']))

    ckpt = torch.load(cfg['model']['load_from'], map_location=f'cuda:{local_rank}')['model']
    from collections import OrderedDict
    _tmp = OrderedDict({k.split('.', 1)[1]:v for k, v in ckpt.items()})
    model.load_state_dict(_tmp)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_output, area_target


def AccuracyCalculate(pred, target, K):
    assert pred.ndim in [1, 2, 3]
    assert pred.shape[0] == target.shape[0]
    area_acc1, _ = np.histogram(pred[:, :1][np.where(pred[:, :1] == target)[0]], bins=np.arange(K + 1))
    area_acc5, _ = np.histogram(pred[np.where(pred == target)], bins=np.arange(K + 1))
    area_pred, _ = np.histogram(pred[:, :1], bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    return area_acc1, area_acc5, area_pred, area_target


def IoU(pred, gt, eval_seg_iou_list):
    cum_I, cum_U, cum_IoU = 0, 0, 0

    if len(pred.shape) == 4:
        pred = (pred > 0.5).squeeze(1).int()

    intersection = torch.sum(torch.mul(pred, gt), dim=(1,2))
    union = torch.sum(torch.add(pred, gt), dim=(1,2)) - intersection
    iou = intersection / (union + 1e-10)

    cum_I += torch.sum(intersection)
    cum_U += torch.sum(union)
    cum_IoU += torch.sum(iou)

    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    for n_eval_iou in range(len(eval_seg_iou_list)):
        eval_seg_iou = eval_seg_iou_list[n_eval_iou]
        seg_correct[n_eval_iou] += torch.sum(iou >= eval_seg_iou)
    seg_total += pred.shape[0]

    return cum_I, cum_U, cum_IoU, seg_correct, seg_total



def metric_table(metrics, metrics_name):
    table = PrettyTable()
    table.field_names = metrics_name
    table.add_row([round(metric, 2) for metric in metrics])

    return table


def make_save_path(args, rank):
    if args.save_path is None:
        if not args.tta:
            args.save_path = args.weight_path.replace('.pth', '_predictions')
        else:
            args.save_path = args.weight_path.replace('.pth', '_tta_predictions')
    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)


def get_save_weight_name(cfg, metric):
    if cfg['model']['backbone']['pretrained'] is None:
        pretrained = 'None'
    elif 'imagenet' in cfg['model']['backbone']['pretrained']:
        pretrained = 'imagenet'
    elif isinstance(cfg['model']['backbone']['pretrained'], list):
        pretrained = 'fullhypersigma'
    else:
        pretrained = (cfg['model']['backbone']['pretrained'].split('/')[-1]).split('.')[0]

    save_name = '.'.join(name for name in [
        cfg['dataset'],
        cfg['model']['backbone']['type'],
        str(cfg.get('model').get('neck').get('type') if cfg.get('model').get('neck') else cfg.get('model').get('neck')),
        str(cfg.get('model').get('decoder').get('type') if cfg.get('model').get('decoder') else cfg.get('model').get('decoder')),
        str(cfg['ngpus']) + 'xb' + str(cfg['batch_size']),
        'img' + str(cfg['crop_size']) if cfg.get('crop_size', False) else '-'.join(str(size) for size in [cfg['data_aug'].get('max_size'), max(cfg['data_aug'].get('scales'))]),
        'ep' + str(cfg['epochs']),
        'pre' + pretrained,
        'best' + str(round(metric, 2)),
        'pth'])

    return save_name


def delete_previous_best_weight(path):
    file_list = os.listdir(path)
    for file_name in file_list:
        if "best" in file_name:
            file_path = os.path.join(path, file_name)
            os.remove(file_path)


def get_dist_info(group: Optional[ProcessGroup] = None) -> Tuple[int, int]:
    if torch_dist.is_available() and torch_dist.is_initialized():
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        rank = torch_dist.get_rank(group)
        world_size = torch_dist.get_world_size(group)
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def txt2json(txt_path, json_path):
    txt_infos = open(txt_path, 'r').readlines()

    image_data = {}
    for txt_info in txt_infos:
        parts = txt_info.strip().split(' ', 5)
        image_name = parts[0]
        bbox = list(map(int, parts[1:5]))
        description = parts[5]

        if image_name not in image_data:
            image_data[image_name] = {'bbox': bbox, 'sentences': []}

        if image_data[image_name]['bbox'] != bbox:
            continue

        image_data[image_name]['sentences'].append(description)

    json.dump(image_data, open(json_path, 'w'))
import argparse, torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.utils import metric_table, make_save_path
from utils.dist_helper import setup_distributed
from dataset.ris import create_test_dataset
from builder import test_model_builder
from engine import eval_ris, predict_ris
# from engine.eval import eval_ris_single_gpu


cudnn.enabled = True
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--task', type=str, default='eval', choices=['eval', 'predict'])
parser.add_argument('--weight-path', type=str, required=True)
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    # initialize
    args = parser.parse_args()
    rank, world_size = setup_distributed(port=args.port)
    # create and load model from weight path
    model, dataset_name, img_size = test_model_builder(args)
    # create test dataset
    testset = create_test_dataset(args.task, dataset_name, 'output_phrase_val', img_size)
    testsampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=1,
                            pin_memory=False, num_workers=8, drop_last=False, shuffle=False)
    # evaluation or prediction
    if args.task == 'eval':
        metrics, metrics_name = eval_ris(model, testloader)
        if rank == 0:
            eval_results = metric_table(metrics, metrics_name)
            print('Evaluation Results: \n{}'.format(eval_results))

    elif args.task == 'predict':
        make_save_path(args, rank)
        predict_ris(model, testloader, args)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()

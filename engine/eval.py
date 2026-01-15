from utils.utils import *
from tqdm import tqdm
import torch.distributed as dist


# def eval_ris_single_gpu(model, loader):
#     # 单卡评估：只在 rank0 调用这个函数
#     model.eval()

#     tbar = tqdm(total=len(loader), desc='Evaluating:')

#     intersection_sum = 0.0
#     union_sum = 0.0
#     iou_sum = 0.0
#     seg_correct_sum = None  # shape: [len(thresholds)]
#     seg_total_sum = 0

#     with torch.no_grad():
#         for img, mask, sentence, attention in loader:
#             img = img.cuda(non_blocking=True)
#             mask = mask.cuda(non_blocking=True)
#             sentence = sentence.cuda(non_blocking=True).squeeze(1)
#             attention = attention.cuda(non_blocking=True).squeeze(1)

#             pred = model(img, sentence, attention).argmax(dim=1)

#             cum_I, cum_U, cum_IoU, seg_correct, seg_total = IoU(
#                 pred, mask, eval_seg_iou_list=[.5, .6, .7, .8, .9]
#             )

#             # cum_I/cum_U/cum_IoU 是 tensor 标量（你的 IoU() 就是 sum）
#             intersection_sum += float(cum_I.item())
#             union_sum += float(cum_U.item())
#             iou_sum += float(cum_IoU.item())

#             if seg_correct_sum is None:
#                 seg_correct_sum = seg_correct.astype(np.float64)
#             else:
#                 seg_correct_sum += seg_correct.astype(np.float64)

#             seg_total_sum += int(seg_total)

#             tbar.update(1)

#     tbar.close()

#     oIoU = intersection_sum / (union_sum + 1e-10) * 100.0
#     mIoU = iou_sum / (seg_total_sum + 1e-10) * 100.0
#     seg_correct_avg = list(seg_correct_sum / (seg_total_sum + 1e-10) * 100.0)
#     seg_correct_avg.append(oIoU)
#     seg_correct_avg.append(mIoU)

#     names = ['PR@.5', 'PR@.6', 'PR@.7', 'PR@.8', 'PR@.9', 'oIoU', 'mIoU']
#     return seg_correct_avg, names

def eval_ris(model, loader):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Evaluating:')

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    iou_meter = AverageMeter()
    seg_correct_meter = AverageMeter()
    seg_total_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, sentence, attention in loader:

            img = img.cuda()
            mask = mask.cuda()
            sentence = sentence.cuda().squeeze(1)
            attention = attention.cuda().squeeze(1)

            pred = model(img, sentence, attention)
            pred = pred.argmax(dim=1)

            cum_I, cum_U, cum_IoU, seg_correct, seg_total = IoU(pred, mask, eval_seg_iou_list = [.5, .6, .7, .8, .9])
            reduced_seg_correct = torch.from_numpy(seg_correct).cuda()
            reduced_seg_total = torch.from_numpy(np.array(seg_total)).cuda()

            dist.all_reduce(cum_I)
            dist.all_reduce(cum_U)
            dist.all_reduce(cum_IoU)
            dist.all_reduce(reduced_seg_correct)
            dist.all_reduce(reduced_seg_total)

            intersection_meter.update(cum_I.cpu().numpy())
            union_meter.update(cum_U.cpu().numpy())
            iou_meter.update(cum_IoU.cpu().numpy())
            seg_correct_meter.update(reduced_seg_correct.cpu().numpy())
            seg_total_meter.update(reduced_seg_total.cpu().numpy())

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()

    oIoU = np.sum(intersection_meter.sum) / (np.sum(union_meter.sum) + 1e-10) * 100
    mIoU = np.sum(iou_meter.sum) / np.sum(seg_total_meter.sum) * 100
    seg_correct_avg = list(seg_correct_meter.sum / seg_total_meter.sum * 100)
    seg_correct_avg.append(oIoU)
    seg_correct_avg.append(mIoU)

    return seg_correct_avg, ['PR@.5', 'PR@.6', 'PR@.7', 'PR@.8', 'PR@.9', 'oIoU', 'mIoU']
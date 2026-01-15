import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import functional as F


def predict_ris(model, loader, args):
    local_rank = int(os.environ['LOCAL_RANK'])

    model.eval()

    if local_rank == 0:
        tbar = tqdm(total=len(loader), desc='Predicting:')

    with torch.no_grad():
        for img, sentence, attention, id in loader:
            img = img.cuda()
            sentence = sentence.cuda().squeeze(1)
            attention = attention.cuda().squeeze(1)

            if args.tta:
                pred = model(img, sentence, attention, args.tta).argmax(dim=1)
            else:
                pred = model(img, sentence, attention).argmax(dim=1)
            for i in range(pred.shape[0]):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                inv_mean = [-m / s for m, s in zip(mean, std)]
                inv_std = [1 / s for s in std]
                im = F.normalize(img, mean=inv_mean, std=inv_std)
                im = im[i, :, :, :].cpu().detach().numpy()  # 注意这里使用i索引当前图像
                im = im.transpose([1, 2, 0])
                im = np.uint8(im * 255)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                output = pred[i].cpu().numpy().astype(np.uint8)  # 转换为uint8
                pred_red_mask = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
                pred_red_mask[output == 1] = (0, 0, 255)  # BGR格式，红色为(0,0,255)
                
                # 生成混合图像
                mixed = cv2.addWeighted(im, 0.5, pred_red_mask, 0.5, 0)
                
                # 创建最终图像，仅混合预测为1的区域
                final_img = im.copy()
                mask = output == 1
                final_img[mask] = mixed[mask]
                
                cv2.imwrite('%s/%s' % (args.save_path, id[i] + '_pred.png'), final_img)

            if local_rank == 0:
                tbar.update(1)

    if local_rank == 0:
        tbar.close()
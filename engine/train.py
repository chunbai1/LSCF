from utils.utils import *
from .eval import eval_ris

def train_ris(args, cfg, model, optimizer, criterion, trainloader, valloader, trainsampler,
              rank, logger, writer, previous_best, lr_scheduler, epoch):
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best oIoU: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()
        trainsampler.set_epoch(epoch)

        for i, (img, mask, sentence, attention) in enumerate(trainloader):
            img, mask = img.cuda(), mask.cuda()
            sentence, attention = sentence.cuda().squeeze(1), attention.cuda().squeeze(1)

            pred = model(img, sentence, attention)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss.update(loss.item())
            iters = epoch * len(trainloader) + i

            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iters)

            if (i % (len(trainloader) // 10) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        metrics, metrics_name = eval_ris(model, valloader)
        # metrics: PR@.5, PR@.6, PR@.7, PR@.8, PR@.9, oIoU, mIoU

        if rank == 0:
            eval_results = metric_table(metrics, metrics_name)
            logger.info('Eval Results: \n{}'.format(eval_results))

            writer.add_scalar('eval/oIoU', metrics[-2], epoch)
            writer.add_scalar('eval/mIoU', metrics[-1], epoch)

        is_best = metrics[-2] > previous_best
        previous_best = max(metrics[-2], previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                save_best_weight_name = get_save_weight_name(cfg, metrics[-2])
                delete_previous_best_weight(args.save_path)
                torch.save(checkpoint, os.path.join(args.save_path, save_best_weight_name))
            if cfg.get('save_interval', -1) > 0:
                if (epoch + 1) % cfg.get('save_interval') == 0:
                    torch.save(checkpoint, os.path.join(args.save_path, 'epoch_{}.pth'.format(epoch + 1)))

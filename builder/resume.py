import os, torch


def resume_training(resume_path, model, optimizer, lr_scheduler, logger):
    checkpoint = torch.load(resume_path, map_location='cpu')

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    previous_best = checkpoint['previous_best']

    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info('************ Resume training from {}'.format(resume_path))

    return model, optimizer, lr_scheduler, epoch, previous_best
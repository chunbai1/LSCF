import torch.nn.functional as F
from torch import nn



def dice_loss(inputs, targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.mean()



def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()



class RISLoss(nn.Module):
    def __init__(self,
                 weight_dice=1.0,
                 weight_focal=1.0,
                 alpha=-1,
                 gamma=0,
                 **kwargs):
        super(RISLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.unsqueeze(1).float()

        loss_dice = dice_loss(inputs, targets)
        loss_focal = sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma)

        total_loss = loss_dice + loss_focal

        return total_loss
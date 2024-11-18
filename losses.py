import torch
import torch.nn as nn
import torch.nn.functional as F




def dice_loss(input, target):
     smooth = 1e-5
     intersect = (input*target).sum(dim=(0, 2, 3))
     union = input.sum(dim=(0,2,3)) + target.sum(dim=(0,2,3))
     return (1 - (intersect + smooth) / (union + smooth)).mean()



def dice(prediction, target):
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = ((prediction[:,i,...]) * target[:,i,...]).sum()
        z_sum = (torch.sum(prediction[:,i,...])).sum()
        y_sum = (torch.sum(target[:,i,...])).sum()
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]
    return loss

def cross_entropy(prediction, target):
    target = target.squeeze(1)
    return nn.CrossEntropyLoss()(prediction, target)


def calc_loss(prediction, target, ce_weight=0.5):

    ce_target = torch.round(target * 3).long()
    ce_target = ce_target.squeeze(1)
    ce_loss = cross_entropy(prediction, ce_target)

    dice_target = F.one_hot(ce_target, num_classes=3)
    dice_target = dice_target.permute(0, 3, 1, 2)
    prediction = torch.softmax(prediction, dim=1)
    dice_l = dice(prediction, dice_target)

    print('dice_loss:{:.6f}, ce_loss:{:.6f}'.format(dice_l, ce_loss))
    return ce_loss * ce_weight + dice_l * (1 - ce_weight)


# def muliclass_dice_score(prediction, target):

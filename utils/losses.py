import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import medpy.metric.binary as mmb
# from pytorch_metric_learning import losses

def calculate_hd95(image1, image2):
    image1_1 = (torch.round(image1*3) >= 1).int().squeeze(0).numpy()
    image2_1 = (torch.round(image2*3) >= 1).int().squeeze(0).numpy()

    image1_2 = (torch.round(image1*3) == 2).int().squeeze(0).numpy()
    image2_2 = (torch.round(image2*3) == 2).int().squeeze(0).numpy()
    hd95_1 = mmb.hd95(image1_1, image2_1)
    hd95_2 = mmb.hd95(image1_2, image2_2)

    return hd95_1, hd95_2


def calculate_assd(image1, image2):
    image1_1 = (torch.round(image1*3) >= 1).int().squeeze(0).numpy()
    image2_1 = (torch.round(image2*3) >= 1).int().squeeze(0).numpy()

    image1_2 = (torch.round(image1*3) == 2).int().squeeze(0).numpy()
    image2_2 = (torch.round(image2*3) == 2).int().squeeze(0).numpy()
    assd_1 = mmb.assd(image1_1, image2_1)
    assd_2 = mmb.assd(image1_2, image2_2)

    return assd_1, assd_2


def dice(prediction, target):
    smooth = 1e-5

    score = 0
    for i in range(target.shape[1]):
        intersect = ((prediction[:,i,...]) * target[:,i,...]).sum()
        z_sum = (torch.sum(prediction[:,i,...]))
        y_sum = (torch.sum(target[:,i,...]))
        score += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    score = score / target.shape[1]
    return score

def cross_entropy(prediction, target):
    target = target.squeeze(1)
    return nn.CrossEntropyLoss()(prediction, target)

# def boundary_loss(prediction, bg_mask, ed_mask, temperature=0.05):
#     bg_features= (prediction * bg_mask).sum(dim=(-1,-2)) / (bg_mask.sum(dim=(-1,-2)))
#     ed_features= (prediction * ed_mask).sum(dim=(-1,-2)) / (ed_mask.sum(dim=(-1,-2)))
#     features = torch.cat([bg_features, ed_features], dim=0)
#     label = torch.cat([torch.zeros(bg_features.shape[0]), torch.ones(ed_features.shape[0])], dim=0)
#     return losses.NTXentLoss(temperature)(features, label)


def calc_loss(prediction, target):

    ce_target = torch.round(target * 3).long()
    ce_target = ce_target.squeeze(1)
    ce_loss = cross_entropy(prediction, ce_target)

    #prediction = F.one_hot(torch.argmax(prediction, dim=1), num_classes=3).permute(0,3,1,2)

    dice_target = F.one_hot(ce_target, num_classes=3).permute(0, 3, 1, 2)
    prediction = torch.softmax(prediction, dim=1)
    dice_loss = 1 - dice(prediction, dice_target)
    return ce_loss, dice_loss

# def muliclass_dice_score(prediction, target):

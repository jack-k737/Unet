from myKits import label_pil2gray, Accumulator
import os
import torchvision.transforms as trans
import PIL.Image as Image
import torch.nn.functional as F
from losses import dice
import torch

predict_image_path = 'model/image_predict/'
true_image_path = 'ODOC/Domain1/test/mask/'

image_list = os.listdir(predict_image_path)
acc = Accumulator(2)
for image in image_list:
    predict_image = Image.open(predict_image_path + image)
    predict_image_t = label_pil2gray(predict_image)
    true_image = Image.open(true_image_path + image)
    true_image_t = label_pil2gray(true_image)

    predict_image_t = F.one_hot(torch.round(predict_image_t*3).long(), num_classes=3).permute(0, 3, 1, 2)
    true_image_t = F.one_hot(torch.round(true_image_t*3).long(), num_classes=3).permute(0, 3, 1, 2)
    dice_score = dice(predict_image_t, true_image_t)
    acc.add(dice_score, 1)

print(acc[0]/acc[1])




#对预测的图像进行评估，计算dice，hd95，assd， 预测的图像位于model/image_predict/，真实的图像位于ODOC/Domain1/test/mask/，计算结果输出到控制台。


from utils.myKits import label_pil2gray, Accumulator
import os
import torchvision.transforms as trans
import PIL.Image as Image
import torch.nn.functional as F
from utils.losses import dice
import torch
from utils.losses import calculate_hd95, calculate_assd
import medpy.metric.binary as mmb


predict_image_path = 'model/image_predict/'
true_image_path = 'ODOC/Domain1/test/mask/'

image_list = os.listdir(predict_image_path)
acc = Accumulator(6)

for image in image_list:
    predict_image = Image.open(predict_image_path + image)
    predict_image_t = label_pil2gray(predict_image)
    true_image = Image.open(true_image_path + image)
    true_image_t = label_pil2gray(true_image)

    true_image_hd95 = true_image_t.squeeze(0)
    predict_image_hd95 = predict_image_t.squeeze(0)
    hd95_1, hd95_2 = calculate_hd95(predict_image_hd95, true_image_hd95)

    assd_1, assd_2 = calculate_assd(predict_image_hd95, true_image_hd95)
    #assd = mmb.assd(predict_image_hd95, true_image_hd95)
    
    predict_image_dice = F.one_hot(torch.round(predict_image_t*3).long(), num_classes=3).permute(0, 3, 1, 2)
    true_image_dice = F.one_hot(torch.round(true_image_t*3).long(), num_classes=3).permute(0, 3, 1, 2)
    dice_score = dice(predict_image_dice, true_image_dice)

    acc.add(dice_score.item(), hd95_1, hd95_2, assd_1, assd_2, 1)
    print('Image:{}  dice -> {:.4f}, hd_95_1 -> {:.4f}, hd_95_2 ->{:.4f}, assd_1 -> {:.4f}, assd_2 -> {:.4f}'.format(image,dice_score.item(), hd95_1, hd95_2, assd_1, assd_2))

    


print('doce -> {:.4f}'.format(acc[0]/acc[5]))
print('hd95 -> {:.4f}, {:.4f}'.format(acc[1]/acc[5], acc[2]/acc[5]))
print('assd -> {:.4f}, {:.4f}'.format(acc[3]/acc[5], acc[4]/acc[5]))





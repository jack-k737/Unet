import PIL.Image as Image
from myKits import label_pil2gray
import torch.nn.functional as F
import torch

if __name__ == '__main__':

    test_image = 'gdrishtiGS_002.png'
    img = Image.open('ODOC/Domain1/train/mask/' + test_image)
    img1 = label_pil2gray(img)
    img2 = torch.round(img1*3).long()
    img3 = F.one_hot(img2, num_classes=3)
    img4 = img3.permute(0, 3, 1, 2)
    print(img.shape)

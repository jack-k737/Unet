import PIL.Image as Image
import torch
import torch.nn as nn

from utils.myKits import label_save_gray, label_pil2gray,create_dir
import os

mask_path = 'ODOC/Domain1/train/mask/'

bgMask_path = 'ODOC/Domain1/train/background_mask/'
edMask_path = 'ODOC/Domain1/train/edge_mask/'

class edgeDetection(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_x.weight = nn.Parameter(torch.tensor([[-1, 0, 1],
                                                         [-2, 0, 2],
                                                         [-1, 0, 1]],dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(torch.tensor([[-1, -2, -1],
                                                         [0,  0,  0],
                                                         [1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        return (torch.sqrt(grad_x ** 2 + grad_y ** 2) > 0.5).float()


if __name__ == '__main__':
    assert os.path.exists(mask_path) and os.path.isdir(mask_path)
    create_dir(bgMask_path, edMask_path)
    mask_list = os.listdir(mask_path)
    for mask_name in mask_list:
        #导入图像，并且装换位灰度图
        mask = Image.open(mask_path + mask_name)
        mask = label_pil2gray(mask)
        #利用
        edge_mask = edgeDetection()(mask)
        edge_mask = edgeDetection()(edge_mask)
        label_save_gray(edge_mask, edMask_path + mask_name)

        background_mask = edgeDetection()(edge_mask)
        background_mask = edgeDetection()(background_mask)
        background_mask = (background_mask.bool() & ~edge_mask.bool()).float()
        label_save_gray(background_mask, bgMask_path + mask_name)


from utils.myKits import freq_space_interpolation
import numpy as np
from glob import glob
import os
import PIL.Image as Image
import torchvision.transforms as transforms 

source_images_dir_mode = os.path.join(os.getcwd(), 'ODOC', 'Domain1','train', 'imgs', '*.png')
source_images_dir = glob(source_images_dir_mode)

target_images_dir_mode = os.path.join(os.getcwd(), 'ODOC', 'Domain[2-5]','train', 'imgs', '*.png')
target_images_dir = glob(target_images_dir_mode)

source_len = len(source_images_dir)
target_len = len(target_images_dir)


for source_image_dir in source_images_dir:
    
    source_image = Image.open(source_image_dir)
    source_image = np.asarray(source_image).transpose((2, 0, 1))


    target_image = Image.open(target_images_dir[np.random.randint(0, target_len)])
    target_image = np.asarray(target_image).transpose((2, 0, 1))
    obj_img_0_2 = freq_space_interpolation(local_img=source_image, target_img=target_image, L=0.1, ratio=0.2).astype(np.uint8)
    obj_img_0_2 = Image.fromarray(np.transpose(obj_img_0_2, (1, 2, 0)))
    obj_img_0_2.save(source_image_dir.replace('imgs', 'freq_interpolation_0_2'))

    target_image = Image.open(target_images_dir[np.random.randint(0, target_len)])
    target_image = np.asarray(target_image).transpose((2, 0, 1))
    obj_img_0_4 = freq_space_interpolation(local_img=source_image, target_img=target_image, L=0.1, ratio=0.4).astype(np.uint8)
    obj_img_0_4 = Image.fromarray(np.transpose(obj_img_0_4, (1, 2, 0)))
    obj_img_0_4.save(source_image_dir.replace('imgs', 'freq_interpolation_0_4'))

    target_image = Image.open(target_images_dir[np.random.randint(0, target_len)])
    target_image = np.asarray(target_image).transpose((2, 0, 1))
    obj_img_0_6 = freq_space_interpolation(local_img=source_image, target_img=target_image, L=0.1, ratio=0.6).astype(np.uint8)
    obj_img_0_6 = Image.fromarray(np.transpose(obj_img_0_6, (1, 2, 0)))
    obj_img_0_6.save(source_image_dir.replace('imgs', 'freq_interpolation_0_6'))

    target_image = Image.open(target_images_dir[np.random.randint(0, target_len)])
    target_image = np.asarray(target_image).transpose((2, 0, 1))
    obj_img_0_8 = freq_space_interpolation(local_img=source_image, target_img=target_image, L=0.1, ratio=0.8).astype(np.uint8)
    obj_img_0_8 = Image.fromarray(np.transpose(obj_img_0_8, (1, 2, 0)))
    obj_img_0_8.save(source_image_dir.replace('imgs', 'freq_interpolation_0_8'))
        
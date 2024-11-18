import PIL.Image as Image
import os
import myKits

label_path  = os.path.join(os.getcwd(),'ODOC','Domain1','test','mask','')
label1_path  = os.path.join(os.getcwd(),'ODOC','Domain1','test','mask1','')
label2_path  = os.path.join(os.getcwd(),'ODOC','Domain1','test','mask2','')
images = os.listdir(label_path)
for image in images:
    img_pil = Image.open(label_path+image)
    img_gray = myKits.label_pil2gray(img_pil)
    img_gray_1 = myKits.image_mask(img_gray, 1 / 3)
    img_gray_2 = myKits.image_mask(img_gray, 2 / 3)
    myKits.label_save_gray(img_gray_1, label1_path + image)
    myKits.label_save_gray(img_gray_2, label2_path + image)

import torch,torchvision
import glob, os, natsort
import PIL.Image as Image

from Models import UNet
from myKits import label_save_gray,create_dir


read_model_path = glob.glob('./model/Unet_*/*.pth')
Test_images = glob.glob('./ODOC/Domain1/test/imgs/*')
image_predict_path = './model/image_predict/'

create_dir(image_predict_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = UNet(3, 3).to(device)
model.load_state_dict(torch.load(read_model_path[0], weights_only=True))


trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


for image_dir in Test_images:
    image = Image.open(image_dir)
    image = trans(image).unsqueeze(0).to(device)
    predict_image, _ = model(image)
    predict_image = torch.argmax(predict_image, dim=1).cpu()
    predict_image = predict_image / 3.
    save_path = image_predict_path + '/' + os.path.basename(image_dir)
    label_save_gray(predict_image, save_path)
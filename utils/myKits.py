import os,shutil
import torch, torchvision
import PIL.Image as Image
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
        self.len = n

    def add(self, *args):
        assert len(args) == self.len
        for i in range(self.len):
            self.data[i] += args[i]

    def reset(self):
        self.data = [0.0] * self.len

    def __getitem__(self, idx):
        return self.data[idx]

def create_dir1(*dir_paths):
    for dir_path in dir_paths:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print('Path: ' + dir_path + ' folder is already existed')
            return
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Creation of the directory '%s' failed" % dir_path)
        else:
            print("Successfully created the directory '%s' " % dir_path)

def create_dir(*dir_paths):
    for dir_path in dir_paths:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            print('Path: ' + dir_path + ' folder there, so deleted for newer one')
        try:
            os.mkdir(dir_path)
        except OSError:
            print("Creation of the directory '%s' failed" % dir_path)
        else:
            print("Successfully created the directory '%s' " % dir_path)

def label_pil2gray(image):
    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Grayscale()(image)
    return image

def label_save_gray(image, path):
    image = image.squeeze(0)
    image_pil = torchvision.transforms.ToPILImage()(image)
    image_pil.save(path)


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.228, 0.225]),
])

def extract_image_features(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT).eval().to(device)
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().cpu().numpy()


def tsne_plot(data, labels_color):
    tsne = TSNE(n_components=3, random_state=0)
    data = tsne.fit_transform(data)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color=labels_color)
    plt.show()


if __name__ == '__main__':
    img = Image.open('ODOC/Domain1/train/imgs/gdrishtiGS_002.png')
    img1 = label_pil2gray(img)
    img2 = torchvision.transforms.ToTensor()(img)
    print(img2.shape)
    # img4 = img3.permute(2,0,1)
    # print(img3.shape)
    # acc = Accumulator(2)
    # acc.add(1, 2)
    # acc.add(3, 4)
    # print(acc[0],acc[1])


from glob import glob
import numpy as np
import os
import PIL.Image as Image

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

label_map = {
    'Domain1': 'r',
    'Domain2': 'g',
    'Domain3': 'b',
    'Domain4': 'y',
    'Domain5': 'c',
}

image_dir_mode = os.path.join(os.getcwd(), 'ODOC','*','*','imgs','*.png')
image_dir_mode1 = os.path.join(os.getcwd(), 'ODOC','Domain1','train','freq_interpolation*','*.png')

image_dirs = glob(image_dir_mode)
image_dirs1 = glob(image_dir_mode1)

images_feature = []
labels = []
colors = []
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
tsne = TSNE(n_components=2, random_state=0)

for image in image_dirs:
    image_feature = np.array(Image.open(image).convert('RGB')).reshape(-1)
    images_feature.append(image_feature)
    labels.append(image.split(os.sep)[-4])
    colors.append(label_map[labels[-1]])

images_feature_np = np.array(images_feature)
data = tsne.fit_transform(images_feature_np)
ax1.scatter(data[:, 0], data[:, 1], color=colors, label = labels)
ax1.set_title('BEFORE SHIFT')

for image in image_dirs1:
    image_feature = np.array(Image.open(image).convert('RGB')).reshape(-1)
    images_feature.append(image_feature)
    labels.append(image.split(os.sep)[-4])
    colors.append(label_map[labels[-1]])

images_feature_np = np.array(images_feature)
data = tsne.fit_transform(images_feature_np)
ax2.scatter(data[:, 0], data[:, 1], color=colors, label = labels)
ax2.set_title('AFTER SHIFT')
plt.tight_layout()
plt.show()




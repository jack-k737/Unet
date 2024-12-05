from utils.myKits import tsne_plot, extract_image_features

from glob import glob
import numpy as np
import os


feature_dir_mode = os.path.join(os.getcwd(), 'ODOC','*','*','imgs','*.npy')
features_dirs = glob(feature_dir_mode)


label_map = {
    'Domain1': 'r',
    'Domain2': 'g',
    'Domain3': 'b',
    'Domain4': 'y',
    'Domain5': 'c',
}

images_feature = []
labels_color = []

for feature_dir in features_dirs:
    images_feature.append(np.load(feature_dir))
    labels_color.append(label_map[feature_dir.split(os.sep)[-4]])

images_feature = np.array(images_feature)

tsne_plot(images_feature, labels_color)


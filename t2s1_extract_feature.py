import os
from glob import glob
import numpy as np
from utils.myKits import extract_image_features

image_dir_mode = os.path.join(os.getcwd(), 'ODOC','*','*','imgs','*.png')
image_dirs = glob(image_dir_mode)


for image_dir in image_dirs:
    image_feature = extract_image_features(image_dir)
    np.save(image_dir.split('.')[0] + '_feature', image_feature)
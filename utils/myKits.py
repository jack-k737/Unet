import os,shutil
import torchvision
import PIL.Image as Image
import numpy as np


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

def amp_spectrum_swap(amp_local, amp_target, L:float= 0.1, ratio:float=0.):
    a_local = np.fft.fftshift(amp_local, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_target, axes=(-2, -1))

    _, h, w = a_local.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b 
    w1 = c_w - b
    w2 = c_w + b

    a_local[:, h1:h2, w1:w2] = a_local[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_local = np.fft.ifftshift(a_local, axes=(-2, -1))


    return a_local


def freq_space_interpolation(local_img, target_img, L:float = 0.1, ratio:float =0):
    local_img_np = local_img

    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1))
    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    fft_target_np = np.fft.fft2(target_img, axes=(-2, -1))
    amp_target = np.abs(fft_target_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap(amp_local, amp_target, L=L, ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp(1j * pha_local)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.clip(np.real(local_in_trg),0,255)


    return local_in_trg


if __name__ == '__main__':
    pass


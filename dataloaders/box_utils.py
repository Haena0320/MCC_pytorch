import os
import random
import numpy as np
import scipy
import warnings
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional
from config import USE_IMAGENET_PRETRAINED

def load_image(img_fn):
    return default_loader(img_fn)  # img --> [H, W, 3]

def resize_image(image, desired_width=768, desired_height=384, random_pad=False):
    w, h = image.size

    width_scale = desired_width / w
    height_scale = desired_height / h
    scale = min(width_scale, height_scale)

    if scale != 1:
        image = functional.resize(image, (round(h * scale), round(w * scale)))

    w, h = image.size
    y_pad = desired_height - h
    x_pad = desired_width - w
    top_pad = random.randint(0, y_pad) if random_pad else y_pad // 2
    left_pad = random.randint(0, x_pad) if random_pad else x_pad // 2

    padding = (left_pad, top_pad, x_pad - left_pad, y_pad - top_pad)
    assert all([x >= 0 for x in padding])
    image = functional.pad(image, padding)
    window = [left_pad, top_pad, w+left_pad, h+top_pad]

    return image, window, scale, padding

if USE_IMAGENET_PRETRAINED:
    def to_tensor_and_normalize(image):
        return functional.normalize(functional.to_tensor(image), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

else:
    def to_tensor_and_normalize(image):
        tensor255 = functional.to_tensor(image) * 255
        return functional.normalize(tensor255, mean=(102.9801, 115.9465, 122.7717), std=(1,1,1))



from torchvision.transforms import functional as FF
import os,sys
import numpy as np
import torch
import random
from PIL import Image,ImageFilter,ImageDraw

#These codes are available at https://github.com/PS06/Copy_Blend.

def apply_data_aug(low_image, high_image, patch_size=48, num_patch=2):
    return copy_blend(low_image, high_image, patch_size, num_patch)
def get_crop_params(img, crop_size):
    w, h = img.size
    if isinstance(crop_size, int):
        th, tw = crop_size, crop_size
    elif isinstance(crop_size, list):
        th = crop_size[0] 
        tw = crop_size[1]
    else:
        raise TypeError

    if w == tw and h == th:
        return 0, 0, h, w
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

def copy_blend(low_image, high_image, patch_size=90, num_patch=2):
    shapes = ['square', 'rect', 'cicle', 'ellipse', 'polygon'] 

    for _ in range(num_patch):
        #idx = random.randrange(0, len(shapes))
        shape = shapes[0]
        i, j, height, width = get_crop_params(low_image, patch_size)
        if shape in ['cicle', 'ellipse', 'polygon']:
            mask = Image.new("L", low_image.size, 255)
            draw = ImageDraw.Draw(mask)

            if shape == 'ellipse':
                draw.ellipse((j, i, height, width), fill=0)
            elif shape == 'cicle':
                radius = max(height, width)
                draw.ellipse((j, i, radius, radius), fill=0)
            else:
                n_sides = random.randint(3, 15)
                draw.regular_polygon((j, i, min(height, width)), n_sides, fill=0)

            low_image = Image.composite(low_image, high_image, mask)            

        else:
            if shape == 'square':
                width = height
            
            low_patch = FF.crop(low_image, i, j, height, width)
            high_patch = FF.crop(high_image, i, j, height, width)
            patch = Image.blend(low_patch, high_patch, random.random())
            low_image.paste(patch, (j, i))
    return low_image

if __name__ == "__main__":
    pass

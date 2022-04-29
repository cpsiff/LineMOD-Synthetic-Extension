import os
from PIL import Image, ImageOps
import numpy as np
import colorsys

"""
https://codegrepr.com/question/changing-image-hue-with-python-pil/
"""
rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

    return new_img

PATH = "lm_test_all/test/000009"
for i in range(len(os.listdir(os.path.join(PATH, "mask")))):
    fname = f"{i:06}"
    rgb_img = Image.open(os.path.join(PATH, 'rgb', fname + ".png"))
    mask_img = ImageOps.invert(Image.open(os.path.join(PATH, 'mask_visib', fname + "_000000.png")))

    blue_img = colorize(rgb_img, 180).convert('RGB') # 50 is about its current hue, 180 is blue, 60 is yellowish

    output_img = Image.composite(rgb_img, blue_img, mask_img).convert('RGB')

    # print("rgb_img.mode", rgb_img.mode)
    # print("mask_img.mode", mask_img.mode)
    # print("blue_img.mode", blue_img.mode)
    # print("output_img.mode", output_img.mode)
    # print("rgb_img.size", rgb_img.size)
    # print("output_img.size", output_img.size)
    # print("rgb_img.info", rgb_img.info)
    # print("output_img.info", output_img.info)
    # print("")

    # for efficientpose, you need to get rid of the first two zeros
    output_img.save(f"real_output/{fname[2:]}.png")

    # for normal linemod, leave them
    # output_img.save(f"real_output/{fname}.png")
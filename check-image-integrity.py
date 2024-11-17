#!/usr/bin/env python

import os
from glob import glob
from tqdm import tqdm

image_ext = ".jxl"

if image_ext == ".jxl":
    import pillow_jxl # noqa: F401
from PIL import Image # noqa: E402
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970

file_list = glob(f"**/*{image_ext}")

for image_path in tqdm(file_list):
    try:
        image = Image.open(image_path)
        image = image.convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        image.close()
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", 'a')
        error_file.write(f"ERROR: {image_path} MESSAGE: {e}\n")
        error_file.close()

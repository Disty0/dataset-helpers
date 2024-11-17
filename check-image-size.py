#!/usr/bin/env python

import os
#import math
from glob import glob
from tqdm import tqdm

image_ext = ".jxl"

if image_ext == ".jxl":
    import pillow_jxl # noqa: F401
from PIL import Image # noqa: E402
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970

file_list = glob(f"**/*{image_ext}")

os.makedirs("out", exist_ok=True)
small_file = open("out/small.txt", 'a')
big_file = open("out/big.txt", 'a')

for image_path in tqdm(file_list):
    try:
        if os.path.getsize(image_path) < 102400:
            #os.remove(image_path)
            small_file.write(image_path+"\n")
        elif os.path.getsize(image_path) > 1024000000:
            image = Image.open(image_path)
            width, height = image.size
            image_size = width * height
            if image_size > 4194304: # 2048x2048
                """
                scale = math.sqrt(image_size / 4194304)
                new_width = int(width/scale)
                new_height = int(height/scale)
                image = image.convert("RGBA").resize((new_width, new_height), Image.LANCZOS)
                image.save(image_path, lossless=True)
                image.close()
                """
                big_file.write(image_path+"\n")
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", 'a')
        error_file.write(f"ERROR: {image_path} MESSAGE: {e}\n")
        error_file.close()
small_file.close()
big_file.close()

#!/usr/bin/env python

import os
import math

from glob import glob
from tqdm import tqdm

import pillow_jxl # noqa: F401
from PIL import Image

remove_files = False
resize_files = False
img_ext_list = ("jpg", "png", "webp", "jpeg")
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


print(f"Searching for {img_ext_list} files...")
file_list = []
for ext in img_ext_list:
    file_list.extend(glob(f"**/*.{ext}"))

os.makedirs("out", exist_ok=True)
resized_jxl_file = open("out/resized_jxl.txt", 'a')

for image_path in tqdm(file_list):
    try:
        base_name, file_ext = os.path.splitext(image_path)
        file_ext = file_ext[1:]
        jxl_path = base_name + ".jxl"

        if not (os.path.exists(jxl_path) and os.path.getsize(jxl_path) != 0):
            image = Image.open(image_path)
            if resize_files and file_ext not in {"jpg", "jpeg"}: # jpeg to lossless jpeg xl has smaller file sizes than resizing down
                width, height = image.size
                image_size = width * height
                if image_size > 4194304:
                    scale = math.sqrt(image_size / 4194304)
                    new_width = int(width/scale)
                    new_height = int(height/scale)
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    resized_jxl_file.write(image_path+"\n")
            image.save(jxl_path, lossless=True, lossless_jpeg=True)
            if remove_files:
                os.remove(image_path)
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", 'a')
        error_file.write(f"ERROR: {image_path} MESSAGE: {e}\n")
        error_file.close()

resized_jxl_file.close()

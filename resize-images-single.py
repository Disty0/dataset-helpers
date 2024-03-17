#!/usr/bin/env python

import os
import gc
import shutil
import glob
from PIL import Image
from tqdm import tqdm

steps_after_gc = 0

def write_caption_to_file(file_name, text):
    caption_file = open(file_name, "r+")
    lines = caption_file.readlines()
    caption_file.seek(0)
    caption_file.write(text)
    for line in lines:
        caption_file.write(line)
    caption_file.close()

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

print("Searching for JPG files...")

for image in tqdm(glob.glob('./*.jpg')):
    try:
        im = Image.open(image)
        im_height, im_width = im.size
        if im_height * im_width > 4194304: # 2048x2048
            im.thumbnail((2048,2048), Image.Resampling.LANCZOS)
            im.save(image, "JPEG", quality=92, optimize=True, subsampling=0)
        im.close()
    except Exception as e:
        print(f"ERROR: {image} MESSAGE: {e}")
        write_caption_to_file("errors/errors.txt", f"\nERROR: {image} MESSAGE: {e}")
        os.makedirs(os.path.dirname(f"errors/{image[2:]}"), exist_ok=True)
        shutil.move(image, f"errors/{image[2:]}")
        try:
            shutil.move(f"{image[:-3]}txt", f"errors/{image[2:-3]}txt")
        except Exception:
            pass
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        gc.collect()
        steps_after_gc = 0


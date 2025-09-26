#!/usr/bin/env python

import os
from glob import glob
from tqdm import tqdm

try:
    import pillow_jxl # noqa: F401
except Exception:
    pass
from PIL import Image # noqa: E402

img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


print(f"Searching for {img_ext_list} files...")
file_list = []
for ext in img_ext_list:
    file_list.extend(glob(f"**/*.{ext}"))

for image_path in tqdm(file_list):
    try:
        image = Image.open(image_path)
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        image.close()
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", "a")
        error_file.write(f"ERROR: {image_path} MESSAGE: {e}\n")
        error_file.close()

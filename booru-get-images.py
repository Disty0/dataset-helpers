#!/usr/bin/env python

import pybooru
import requests
from PIL import Image
from tqdm import tqdm

pybooru.resources.SITE_LIST["shima"] = {"url": "https://shima.donmai.us/"}
client = pybooru.Danbooru('shima')

for id in tqdm(range(7309999, 7310001)):
    try:
        image_data = client.post_show(id)
        image = Image.open(requests.get(image_data["file_url"], stream=True).raw).convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255))
        new_image = Image.alpha_composite(background, image).convert("RGB")
        im_height, im_width = new_image.size
        if im_height * im_width > 4194304: # 2048x2048
            new_image.thumbnail((2048,2048), Image.Resampling.LANCZOS)
        new_image.save(f"{id}.jpg", "JPEG", quality=92, optimize=True, subsampling=0)
        new_image.close()
        image.close()
        caption_file = open(f"{id}.txt", "w")
        caption_file.write(image_data["tag_string"].replace(" ", ", ").replace("_", " "))
        caption_file.close()
    except Exception as e:
        pass


#!/usr/bin/env python

import os
import math
import json
import time
import argparse
import pybooru
import requests
from PIL import Image
from tqdm import tqdm

pybooru.resources.SITE_LIST["shima"] = {"url": "https://shima.donmai.us/"}
client = pybooru.Danbooru('shima')

parser = argparse.ArgumentParser(description='Get images from danbooru')
parser.add_argument('start', type=int)
parser.add_argument('end', type=int)
args = parser.parse_args()

general_blacklist = [
    "comic",
    "text_focus",
    "realistic",
    "what",
    "misplaced_genitals",
    "anatomical_nonsense",
    "engineering_nonsense",
    "artistic_error",
    "extra_digits",
    "wrong_foot",
    "wrong_hand",
    "small_hands",
    "bad_reflection",
    "bad_multiple_views",
    "bad_anatomy",
    "bad_vehicle_anatomy"
    "bad_gun_anatomy"
    "bad_proportions",
    "bad_face",
    "bad_hands",
    "bad_leg",
    "bad_feet",
    "bad_ass",
    "bad_vulva",
    "bad_neck",
]

meta_blacklist = [
    "artifacts",
    "aliasing",
    "adversarial_noise",
    "comic",
    "crease",
    "duplicate",
    "lowres",
    "sample",
    "color_halftone",
    "color_issue",
    "scan_dust",
    "bleed_through",
    "binding_discoloration",
    "bad_aspect_ratio",
    "poorly_stitched",
    "photo_(medium)",
]


for id in tqdm(range(args.start, args.end)):
    try:
        folder = str(int(id / 10000))
        os.makedirs(folder, exist_ok=True)
        image_data = client.post_show(id)
        width = int(image_data["image_width"])
        height = int(image_data["image_height"])
        image_size = width * height
        general_tags = image_data["tag_string_general"].split(" ")
        if (image_data["file_ext"] not in {"avi", "gif", "html", "mp3", "mp4", "mpg", "pdf", "rar", "swf", "webm", "wmv", "zip"}
        and image_size > 768000
        and not image_data["is_banned"]
        and not image_data["is_flagged"]
        and not image_data["is_deleted"]
        and not any([bool(tag in general_tags) for tag in general_blacklist])
        and not any([bool(tag in image_data["tag_string_meta"]) for tag in meta_blacklist])):
            image = Image.open(requests.get(image_data["file_url"], stream=True).raw).convert('RGBA')
            if image_size > 4194304: # 2048x2048
                scale = math.sqrt(image_size / 4194304)
                new_width = int(width/scale)
                new_height = int(height/scale)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            image.save(os.path.join(folder, f"{id}.webp"), "WEBP", quality=99)
            image.close()
            with open(os.path.join(folder, f"{id}.json"), "w") as f:
                json.dump(image_data, f)
        else:
            time.sleep(0.25)
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open(f"errors/errors{args.start}.txt", 'a')
        error_file.write(f"ERROR: {id} MESSAGE: {e} \n")
        error_file.close()



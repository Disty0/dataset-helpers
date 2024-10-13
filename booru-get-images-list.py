#!/usr/bin/env python

import os
import math
import json
import time
import argparse
import pybooru
import requests
from tqdm import tqdm

image_ext = ".jxl"
pybooru.resources.SITE_LIST["shima"] = {"url": "https://shima.donmai.us/"}
client = pybooru.Danbooru('shima')

if image_ext == ".jxl":
    import pillow_jxl # noqa: F401
from PIL import Image # noqa: E402
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970

parser = argparse.ArgumentParser(description='Get images from danbooru')
parser.add_argument('list', type=str)

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


with open(args.list, "r") as file:
    id_list = file.readlines()

new_id_list = []
for id in id_list:
    if id:
        new_id_list.append(int(id))
id_list = new_id_list

for id in tqdm(id_list):
    folder = str(int(id / 10000))
    image_path = os.path.join(folder, str(id)+image_ext)

    if not (os.path.exists(image_path) and os.path.getsize(image_path) != 0):
        json_path = os.path.join(folder, f"{id}.json")
        jpg_path = None
    
        if os.path.exists(json_path) and os.path.getsize(json_path) != 0:
            with open(json_path, "r") as f:
                image_data = json.load(f)
        else:
            try:
                time.sleep(0.25)
                image_data = client.post_show(id)
                os.makedirs(folder, exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump(image_data, f)
            except Exception as e:
                str_e = str(e)
                if not ("In _request: 404 - Not Found" in str_e and str_e.endswith(".json")):
                    os.makedirs("errors", exist_ok=True)
                    error_file = open("errors/errors_json_list.txt", 'a')
                    error_file.write(f"ERROR: {id} MESSAGE: {str_e}\n")
                    error_file.close()
                continue

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
            try:
                if image_ext == ".jxl" and image_data["file_ext"] in {"jpg", "jpeg"}:
                    jpg_path = os.path.join(folder, str(id)+".jpg")
                    image_data = requests.get(image_data["file_url"], stream=True).raw.read()
                    with open(jpg_path, "wb") as jpg_file:
                        jpg_file.write(image_data)
                    image = Image.open(jpg_path)
                else:
                    image = Image.open(requests.get(image_data["file_url"], stream=True).raw).convert("RGBA")
                    if image_size > 4194304: # 2048x2048
                        scale = math.sqrt(image_size / 4194304)
                        new_width = int(width/scale)
                        new_height = int(height/scale)
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                image.save(image_path, lossless=True)
                image.close()
                if jpg_path is not None and os.path.exists(jpg_path):
                    os.remove(jpg_path)
            except Exception as e:
                if jpg_path is not None and os.path.exists(jpg_path):
                    os.remove(jpg_path)
                str_e = str(e)
                if str_e != "'file_url'":
                    os.makedirs("errors", exist_ok=True)
                    error_file = open("errors/errors_list.txt", 'a')
                    error_file.write(f"ERROR: {id} MESSAGE: {str_e}\n")
                    error_file.close()

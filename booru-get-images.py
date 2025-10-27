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
client = pybooru.Danbooru("shima")

if image_ext == ".jxl":
    import pillow_jxl # noqa: F401
from PIL import Image # noqa: E402
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970

parser = argparse.ArgumentParser(description="Get images from danbooru")
parser.add_argument("start", type=int)
parser.add_argument("end", type=int)
args = parser.parse_args()

general_blacklist = (
    "what",
    "comic",
    "text_focus",
    "realistic",
    "photorealistic",
    "misplaced_genitals",
    "anatomical_nonsense",
    "engineering_nonsense",
    "artistic_error",
    "wrong_foot",
    "wrong_hand",
    "small_hands",
    "body_horror",
    "conjoined",
    "bad_reflection",
    "bad_multiple_views",
    "bad_anatomy",
    "bad_internal_anatomy",
    "bad_vehicle_anatomy",
    "bad_gun_anatomy",
    "bad_proportions",
    "bad_perspective",
    "bad_bowmanship",
    "bad_weapon",
    "bad_shadow",
    "bad_knees",
    "bad_face",
    "bad_teeth",
    "bad_hands",
    "bad_leg",
    "bad_feet",
    "bad_arm",
    "bad_ass",
    "bad_vulva",
    "bad_neck",
    "extra_digits",
    "extra_pussies",
    "extra_breasts",
    "extra_nipples",
    "extra_penises",
    "extra_faces",
    "extra_legs",
    "extra_hands",
    "extra_tails",
    "extra_horns",
    "extra_tusks",
    "extra_noses",
    "extra_eyelids",
    "extra_testicles",
    "extra_clitorises",
    "extra_chest",
    "extra_eyewear",
)

meta_blacklist = (
    "off-topic",
    "corrupted_file",
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
)


for id in tqdm(range(args.start, args.end)):
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
                    error_file = open(f"errors/errors_json{args.start}.txt", "a")
                    error_file.write(f"ERROR: {id} MESSAGE: {str_e}\n")
                    error_file.close()
                continue

        if image_data.get("file_url", None) is not None:
            width = int(image_data["image_width"])
            height = int(image_data["image_height"])
            image_size = width * height
            general_tags = image_data["tag_string_general"].split(" ")
            general_tags.extend(image_data.get("wd_tag_string_general", "").split(" "))
            is_pixel_art = "pixel_art" in general_tags
        else:
            continue

        if (
            image_data["score"] > 0
            and not image_data["is_banned"]
            and not image_data["is_flagged"]
            and not image_data["is_deleted"]
            and (image_size > 768000 or is_pixel_art)
            and (image_data["file_size"] > 102400 or is_pixel_art)
            and image_data["file_ext"] not in {"avif", "avi", "gif", "html", "mp3", "mp4", "mpg", "pdf", "rar", "swf", "webm", "wmv", "zip"}
            and not any([bool(tag in general_tags) for tag in general_blacklist])
            and not any([bool(tag in image_data["tag_string_meta"]) for tag in meta_blacklist])
        ):
            try:
                image = requests.get(image_data["file_url"], stream=True).raw
                if image_ext == ".jxl" and image_data["file_ext"] in {"jpg", "jpeg"}:
                    jpg_path = os.path.join(folder, str(id)+".jpg")
                    with open(jpg_path, "wb") as jpg_file:
                        jpg_file.write(image.read())
                    image = Image.open(jpg_path)
                else:
                    image = Image.open(image).convert("RGBA")
                    if image_size > 4194304: # 2048x2048
                        scale = math.sqrt(image_size / 4194304)
                        new_width = int(width/scale)
                        new_height = int(height/scale)
                        image = image.resize((new_width, new_height), Image.LANCZOS)
                if is_pixel_art:
                    width, height = image.size
                    image_size = width * height
                    if image_size < 1048576:
                        scale = math.sqrt(image_size / 1048576)
                        new_width = int(width/scale)
                        new_height = int(height/scale)
                        image = image.convert("RGBA").resize((new_width, new_height), Image.NEAREST)
                try:
                    image.save(image_path, lossless=True, lossless_jpeg=True)
                except Exception as e:
                    if jpg_path is not None:
                        image = image.convert("RGBA")
                        if image_size > 4194304: # 2048x2048
                            scale = math.sqrt(image_size / 4194304)
                            new_width = int(width/scale)
                            new_height = int(height/scale)
                            image = image.resize((new_width, new_height), Image.LANCZOS)
                        image.save(image_path, lossless=True, lossless_jpeg=False)
                    else:
                        raise e
                image.close()
                if jpg_path is not None and os.path.exists(jpg_path):
                    os.remove(jpg_path)
            except Exception as e:
                if jpg_path is not None and os.path.exists(jpg_path):
                    os.remove(jpg_path)
                str_e = str(e)
                if str_e != "'file_url'":
                    os.makedirs("errors", exist_ok=True)
                    error_file = open(f"errors/errors{args.start}.txt", "a")
                    error_file.write(f"ERROR: {id} MESSAGE: {str_e}\n")
                    error_file.close()

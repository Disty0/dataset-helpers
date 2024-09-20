#!/usr/bin/env python

import os
import glob
import json
from tqdm import tqdm

image_ext = ".webp"

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


os.makedirs("out", exist_ok=True)
out_file = open("out/to_purge.txt", "a")

print("Searching for JSON files...")
file_list = glob.glob('./**/*.json')

for json_path in tqdm(file_list):
    try:
        if os.path.exists(os.path.splitext(json_path)[0]+image_ext):
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)
            general_tags = json_data["tag_string_general"].split(" ")
            if (json_data["is_banned"]
            or json_data["is_flagged"]
            or json_data["is_deleted"]
            or any([bool(tag in general_tags) for tag in general_blacklist])
            or any([bool(tag in json_data["tag_string_meta"]) for tag in meta_blacklist])):
                out_file.write(json_path[2:-5] + "\n")
    except Exception as e:
        print(f"ERROR: {json_path} MESSAGE: {e}")
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors_purge.txt", 'a')
        error_file.write(f"ERROR: {json_path} MESSAGE: {e}")
        error_file.close()

#!/usr/bin/env python

import os
import json
from glob import glob
from tqdm import tqdm

remove_files = False
img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
bad_image_count = 0

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

os.makedirs("out", exist_ok=True)
out_file = open("out/tag_check.txt", "a")

print(f"Searching for {img_ext_list} files...")
file_list = []
for ext in img_ext_list:
    file_list.extend(glob(f"**/*.{ext}"))

do_write = True
progress_bar = tqdm(file_list)
for image_path in file_list:
    try:
        base_name, file_ext = os.path.splitext(image_path)
        file_ext = file_ext[1:]
        json_path = base_name+".json"

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        general_tags = json_data["tag_string_general"].split(" ")
        general_tags.extend(json_data.get("wd_tag_string_general", "").split(" "))

        if (
            json_data.get("is_banned", False) or json_data.get("is_flagged", False) or json_data.get("is_deleted", False)
            or any([bool(tag in general_tags) for tag in general_blacklist])
            or any([bool(tag in json_data.get("tag_string_meta", "")) for tag in meta_blacklist])
        ):
            bad_image_count += 1
            if remove_files:
                os.remove(image_path)
            out_file.write(image_path+"\n")
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", "a")
        error_file.write(f"ERROR: {image_path} MESSAGE: {e}\n")
        error_file.close()
    progress_bar.set_postfix(cleanup=bad_image_count)
    progress_bar.update(1)

out_file.close()

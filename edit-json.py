#!/usr/bin/env python

import os
import json
from glob import glob
from tqdm import tqdm

img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")

key_to_remove_list = (
    #"gemma-3n-e4b-it",
    #"gemma-3-4b-it",
    #"gemma-3-12b-it",
    #"gemma-3-27b-it",
    #"qwen3-vl-8b-nsfw-caption-v4",
    #"qwen3-vl-8b-instruct",
    #"qwen3-vl-32b-instruct",
    #"qwen3-vl-32b-instruct-sdnq-uint4-svd-r32",
    #"florence-2-base-promptgen-v1-5",
)

total_cleanup = 0

os.makedirs("out", exist_ok=True)
out_file = open("out/out_edit.txt", "a")

print(f"Searching for {img_ext_list} files...")
file_list = []
for ext in img_ext_list:
    file_list.extend(glob(f"**/*.{ext}"))

progress_bar = tqdm(file_list)
for image_path in file_list:
    try:
        do_write = False
        base_name, file_ext = os.path.splitext(image_path)
        file_ext = file_ext[1:]
        json_path = base_name+".json"

        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

        if json_data.get("file_ext", None) is None:
            json_data["file_ext"] = file_ext
            do_write = True
        if json_data.get("file_size", None) is None:
            json_data["file_size"] = os.path.getsize(image_path)
            do_write = True

        for key in key_to_remove_list:
            key_data = json_data.pop(key, None)
            if key_data is not None:
                do_write = True

        if do_write:
            total_cleanup += 1
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file)
            out_file.write(json_path+"\n")
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", "a")
        error_file.write(f"ERROR: {image_path} MESSAGE: {e}\n")
        error_file.close()
    progress_bar.set_postfix(cleanup=total_cleanup)
    progress_bar.update(1)

out_file.close()

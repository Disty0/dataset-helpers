#!/usr/bin/env python

import os
import shutil
import glob
import random
from tqdm import tqdm

count = 0
shard = 0
count_shard = 0

def write_caption_to_file(file_name, text):
    caption_file = open(file_name, "r+")
    lines = caption_file.readlines()
    caption_file.seek(0)
    caption_file.write(text)
    for line in lines:
        caption_file.write(line)
    caption_file.close()

print("Searching for TXT files...")
file_list = glob.glob('./**/*.txt')
random.shuffle(file_list)

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
try:
    shutil.move("../moved_dataset", "../moved_dataset_old")
except Exception:
    pass
os.makedirs(os.path.dirname(f"../moved_dataset/{str(shard).zfill(4)}/"), exist_ok=True)
open("errors/errors.txt", 'a').close()

for txt_file in tqdm(file_list):
    try:
        shutil.move(txt_file[:-3]+"jpg", f"../moved_dataset/{str(shard).zfill(4)}/{str(count).zfill(8)}.jpg")
        shutil.move(txt_file, f"../moved_dataset/{str(shard).zfill(4)}/{str(count).zfill(8)}.txt")
        count = count + 1
        count_shard = count_shard + 1
        if count_shard >= 8000:
            shard = shard + 1
            os.makedirs(os.path.dirname(f"../moved_dataset/{str(shard).zfill(4)}/"), exist_ok=True)
            count_shard = 0
    except Exception as e:
        print(f"ERROR: {txt_file} MESSAGE: {e}")
        write_caption_to_file("errors/errors.txt", f"\nERROR: {txt_file} MESSAGE: {e}")
        os.makedirs(os.path.dirname(f"errors/{txt_file[2:]}"), exist_ok=True)
        shutil.move(txt_file, f"errors/{txt_file[2:]}")
        try:
            shutil.move(f"{txt_file[:-3]}jpg", f"errors/{txt_file[2:-3]}jpg")
        except Exception:
            pass


#!/usr/bin/env python

import os
import gc
import shutil
import glob

from tqdm import tqdm


steps_after_gc = 0

def write_caption_to_file(file_name):
    caption_file = open(file_name, "r")
    line = caption_file.readlines()[0].split(", ")
    caption_file.close()
    caption_file = open(file_name, "w")
    caption_file.seek(0)
    line = f"{line[0]}, {line[1]}, {line[2]}, from {line[3]}"
    caption_file.write(line)
    caption_file.close()
    

print("Searching for TXT files...")
file_list = glob.glob('./**/*.txt')

for text_path in tqdm(file_list):
    try:
        write_caption_to_file(text_path)
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors_aesthetic.txt", 'a')
        error_file.write(f"ERROR: {text_path} MESSAGE: {e} \n")
        error_file.close()
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        gc.collect()
        steps_after_gc = 0


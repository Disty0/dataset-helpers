#!/usr/bin/env python

import gc
import glob

from tqdm import tqdm


steps_after_gc = 0

def write_caption_to_file(file_name):
    caption_file = open(file_name, "r")
    line = caption_file.readlines()[0].rsplit(", ")
    caption_file.close()
    caption_file = open(file_name, "w")
    caption_file.seek(0)
    line = f"{line[0]}, {line[1]}, {line[2]},"
    caption_file.write(line)
    caption_file.close()
    

print("Searching for TXT files...")
file_list = glob.glob('./**/*.txt')

for text in tqdm(file_list):
    try:
        write_caption_to_file(text)
    except Exception as e:
        print(f"ERROR: {text} MESSAGE: {e}")
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        gc.collect()
        steps_after_gc = 0


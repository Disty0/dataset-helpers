#!/usr/bin/env python

import os
import gc
import shutil
import glob

from tqdm import tqdm


steps_after_gc = 0

def remove_old_tag(text):
    text = text.removeprefix("out of the scale quality, ")
    text = text.removeprefix("masterpiece, ")
    text = text.removeprefix("best quality, ")
    text = text.removeprefix("high quality, ")
    text = text.removeprefix("great quality, ")
    text = text.removeprefix("medium quality, ")
    text = text.removeprefix("normal quality, ")
    text = text.removeprefix("bad quality, ")
    text = text.removeprefix("low quality, ")
    text = text.removeprefix("worst quality, ")
    return text

def write_caption_to_file(file_name):
    caption_file = open(file_name, "r")
    line = caption_file.readlines()[0]
    caption_file.close()
    caption_file = open(file_name, "w")
    caption_file.seek(0)
    line = remove_old_tag(line)
    caption_file.write(line)
    caption_file.close()
    

print("Searching for TXT files...")
file_list = glob.glob('./*.txt')

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

for text in tqdm(file_list):
    try:
        write_caption_to_file(text)
    except Exception as e:
        print(f"ERROR: {text} MESSAGE: {e}")
        write_caption_to_file("errors/errors.txt", f"\nERROR: {text} MESSAGE: {e}")
        os.makedirs(os.path.dirname(f"errors/{text[2:]}"), exist_ok=True)
        shutil.move(text, f"errors/{text[2:]}")
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        gc.collect()
        steps_after_gc = 0


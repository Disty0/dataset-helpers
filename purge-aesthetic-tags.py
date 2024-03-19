#!/usr/bin/env python

import os
import gc
import shutil
import glob

from tqdm import tqdm


steps_after_gc = 0

def remove_old_tag(text):
    text = text.removeprefix("out of the scale aesthetic, ")
    text = text.removeprefix("masterpiece, ")
    text = text.removeprefix("extremely aesthetic, ")
    text = text.removeprefix("very aesthetic, ")
    text = text.removeprefix("asthetic, ")
    text = text.removeprefix("aesthetic, ")
    text = text.removeprefix("slightly asthetic, ")
    text = text.removeprefix("slightly aesthetic, ")
    text = text.removeprefix("not displeasing, ")
    text = text.removeprefix("not asthetic, ")
    text = text.removeprefix("not aesthetic, ")
    text = text.removeprefix("slightly displeasing, ")
    text = text.removeprefix("displeasing, ")
    text = text.removeprefix("very displeasing, ")
    return text

def write_caption_to_file(file_name):
    caption_file = open(file_name, "r+")
    lines = caption_file.readlines()
    caption_file = open(file_name, "r+")
    caption_file.close()
    caption_file = open(file_name, "w")
    caption_file.seek(0)
    for line in lines:
        line = remove_old_tag(line)
        caption_file.write(line)
    caption_file.close()
    

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

print("Searching for TXT files...")

for text in tqdm(glob.glob('./*.txt')):
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


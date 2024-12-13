#!/usr/bin/env python

import os
import gc
import glob

from tqdm import tqdm

steps_after_gc = 0

def write_tag(file_name):
    global tags_list, tags_file_list, words_list, txt_list
    caption_file = open(file_name, "r")
    tags = caption_file.readlines()[0].replace("\n", "")
    caption_file.close()
    tags_file_list.write(f"{file_name}  :::  {tags}\n")
    for tag in tags.split(", "):
        if tag:
            tags_list.write(tag + "\n")
    for word in tags.replace(",", "").replace(".", "").replace("\n", " ").split(" "):
        if word:
            words_list.write(word + "\n")


print("Searching for TXT files...")
file_list = glob.glob('**/*.txt')

os.makedirs("out", exist_ok=True)

tags_list = open("out/tags_list.txt", 'w')
tags_file_list = open("out/tags_file_list.txt", 'w')
words_list = open("out/words_list.txt", 'w')

for text_path in tqdm(file_list):
    try:
        write_tag(text_path)
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", 'a')
        error_file.write(f"ERROR: {text_path} MESSAGE: {e} \n")
        error_file.close()
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 100000:
        gc.collect()
        steps_after_gc = 0


tags_list.close()
tags_file_list.close()
words_list.close()

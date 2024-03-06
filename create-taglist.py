#!/usr/bin/env python

import os
import gc
import glob

from tqdm import tqdm

steps_after_gc = 0

def write_error(text):
    caption_file = open("errors/errors.txt", "r+")
    lines = caption_file.readlines()
    caption_file.seek(0)
    caption_file.write(text)
    for line in lines:
        caption_file.write(line)
    caption_file.close()


def write_tag(file_name):
    global tags_list, tags_file_list, words_list, txt_list
    caption_file = open(file_name, "r+")
    tags = caption_file.readlines()[0].replace("\n", "")
    txt_list.write(file_name + "\n")
    tags_file_list.write(f"{file_name}  :::  {tags}\n")
    for tag in tags.rsplit(", "):
        tags_list.write(tag + "\n")
    for word in tags.replace(",", "").rsplit(" "):
        words_list.write(word + "\n")

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
os.makedirs(os.path.dirname("out/txt_list.txt"), exist_ok=True)
os.makedirs(os.path.dirname("out/tags_list.txt"), exist_ok=True)
os.makedirs(os.path.dirname("out/tags_file_list.txt"), exist_ok=True)
os.makedirs(os.path.dirname("out/words_list.txt"), exist_ok=True)

open("errors/errors.txt", 'a').close()
txt_list = open("out/txt_list.txt", 'w')
tags_list = open("out/tags_list.txt", 'w')
tags_file_list = open("out/tags_file_list.txt", 'w')
words_list = open("out/words_list.txt", 'w')

print("Searching for TXT files...")

for text in tqdm(glob.glob('./**/**/*.txt')):
    try:
        write_tag(text)
    except Exception as e:
        print(f"ERROR: {text} MESSAGE: {e}")
        write_error(f"\nERROR: {text} MESSAGE: {e}")
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 100000:
        gc.collect()
        steps_after_gc = 0


txt_list.close()
tags_list.close()
tags_file_list.close()
words_list.close()

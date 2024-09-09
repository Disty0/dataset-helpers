#!/usr/bin/env python

import os
import gc
import glob

from tqdm import tqdm

tag_list = [
    [", questionable, ", ", nsfw, "],
    [", explicit, ", ", explicit nsfw, "],
]

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
    global tag_list
    caption_file = open(file_name, "r+")
    #new_file_name = "/home/disty/dataset" + os.getcwd() +  file_name[1:]
    #os.makedirs(os.path.dirname(new_file_name), exist_ok=True)
    #caption_file_2 = open(new_file_name, "w")
    tags = caption_file.readlines()[0]
    caption_file.close()
    caption_file_2 = open(file_name, "w")
    for old_tag, new_tag in tag_list:
        tags = tags.replace(old_tag, new_tag)
    while tags[0] == " ":
        tags = tags[1:]
    while tags[-1] == " ":
        tags = tags[:-1]
    while tags[-1] == ".":
        tags = tags[:-1]
    if tags[:9] == "an  girl ":
        tags = "a girl " + tags[9:]
    caption_file_2.write(tags)
    caption_file_2.close()

print("Searching for TXT files...")
file_list = glob.glob('./**/*.txt')

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

for text in tqdm(file_list):
    try:
        write_tag(text)
    except Exception as e:
        print(f"ERROR: {text} MESSAGE: {e}")
        write_error(f"\nERROR: {text} MESSAGE: {e}")
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 100000:
        gc.collect()
        steps_after_gc = 0


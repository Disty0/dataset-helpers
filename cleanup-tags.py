#!/usr/bin/env python

import os
import gc
import glob

from tqdm import tqdm

tag_list = [
    [", questionable, ", ", nsfw, "],
    [", explicit, ", ", explicit nsfw, "],
]

rating_list = [
    [", general", ", sfw"],
    [", sensitive", ", suggestive"],
    [", questionable", ", nsfw"],
    [", explicit", ", explicit nsfw"],
]

steps_after_gc = 0

def cleanup_whitespace(caption):
    while caption[0] == ",":
        caption = caption[1:]
    while caption[0] == ".":
        caption = caption[1:]
    while caption[0] == " ":
        caption = caption[1:]
    while caption[-1] == " ":
        caption = caption[:-1]
    while caption[-1] == ".":
        caption = caption[:-1]
    while caption[-1] == ",":
        caption = caption[:-1]
    return caption


def write_tag(file_name):
    global tag_list
    caption_file = open(file_name, "r+")
    tags = caption_file.readlines()[0].replace("\n", "")
    caption_file.close()
    caption_file_2 = open(file_name, "w")
    cleanup_whitespace(tags)
    #for old_tag, new_tag in tag_list:
    #    tags = tags.replace(old_tag, new_tag)
    for old_tag, new_tag in rating_list:
        if tags[-len(old_tag):] == old_tag:
            tags = tags[:-len(old_tag)] + new_tag
    cleanup_whitespace(tags)
    caption_file_2.write(tags)
    caption_file_2.close()

print("Searching for TXT files...")
file_list = glob.glob('**/*.txt')


for txt_path in tqdm(file_list):
    try:
        write_tag(txt_path)
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", 'a')
        error_file.write(f"ERROR: {txt_path} MESSAGE: {e} \n")
        error_file.close()
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 100000:
        gc.collect()
        steps_after_gc = 0


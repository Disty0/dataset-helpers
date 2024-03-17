#!/usr/bin/env python

import os
import gc
import shutil
import glob

from tqdm import tqdm

steps_after_gc = 0

def write_error(text):
    global error_list
    error_list.write("\n" + text)

def write_tag(file_name):
    global tag_list
    caption_file = open(file_name, "r+")
    tag = caption_file.readlines()[0].rsplit(", ")[-1]
    caption_file.close()
    if tag[-3:-1] == tag[-5:-3]:
        tag_list_2.write("\n" + tag)
    if tag[-4:-1] == tag[-7:-4]:
        tag_list_3.write("\n" + tag)


os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
os.makedirs(os.path.dirname("out/tag_list_2.txt"), exist_ok=True)
os.makedirs(os.path.dirname("out/tag_list_3.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()
open("out/tag_list_2.txt", 'a').close()
open("out/tag_list_3.txt", 'a').close()
error_list = open("errors/errors.txt", "r+")
tag_list_2 = open("out/tag_list_2.txt", "r+")
tag_list_3 = open("out/tag_list_3.txt", "r+")

print("Searching for TXT files...")

for text in tqdm(glob.glob('./**/*.txt')):
    try:
        write_tag(text)
    except Exception as e:
        print(f"ERROR: {text} MESSAGE: {e}")
        write_error(f"\nERROR: {text} MESSAGE: {e}")
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        gc.collect()
        steps_after_gc = 0


#!/usr/bin/env python

import os
import gc
import shutil
import glob
import torch

from transformers import pipeline
from tqdm import tqdm

pipe = pipeline("image-classification", model="shadowlilac/aesthetic-shadow", device="cuda")
steps_after_gc = 0

def write_caption_to_file(file_name, text):
    caption_file = open(file_name, "r+")
    lines = caption_file.readlines()
    caption_file.seek(0)
    caption_file.write(text)
    for line in lines:
        caption_file.write(line)
    caption_file.close()
    
def write_caption(file_name, score):
    if score > 1.50: # out of the scale
        write_caption_to_file(file_name, "out of the scale aesthetic, ")
    if score > 1.10: # out of the scale
        write_caption_to_file(file_name, "masterpiece, ")
    elif score > 0.98:
        write_caption_to_file(file_name, "extremely aesthetic, ")
    elif score > 0.90:
        write_caption_to_file(file_name, "very aesthetic, ")
    elif score > 0.75:
        write_caption_to_file(file_name, "asthetic, ")
    elif score > 0.50:
        write_caption_to_file(file_name, "slightly asthetic, ")
    elif score > 0.35:
        write_caption_to_file(file_name, "not displeasing, ")
    elif score > 0.25:
        write_caption_to_file(file_name, "not asthetic, ")
    elif score > 0.125:
        write_caption_to_file(file_name, "slightly displeasing, ")
    elif score > 0.025:
        write_caption_to_file(file_name, "displeasing, ")
    else:
        write_caption_to_file(file_name, "very displeasing, ")

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

print("Searching for JPG files...")

for image in tqdm(glob.glob('./*.jpg')):
    try:
        prediction_single = pipe(images=[image])[0]
        if prediction_single[0]["label"] == "hq":
            prediction = prediction_single[0]["score"]
        elif prediction_single[1]["label"] == "hq":
            prediction = prediction_single[1]["score"]
        write_caption(image[:-3]+"txt", prediction)
    except Exception as e:
        print(f"ERROR: {image} MESSAGE: {e}")
        write_caption_to_file("errors/errors.txt", f"\nERROR: {image} MESSAGE: {e}")
        os.makedirs(os.path.dirname(f"errors/{image[2:]}"), exist_ok=True)
        shutil.move(image, f"errors/{image[2:]}")
        try:
            shutil.move(f"{image[:-3]}txt", f"errors/{image[2:-3]}txt")
        except Exception:
            pass
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        steps_after_gc = 0


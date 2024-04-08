#!/usr/bin/env python

import os
import gc
import shutil
import glob
import torch
try:
    import intel_extension_for_pytorch as ipex
except Exception:
    pass

from transformers import pipeline
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
pipe = pipeline("image-classification", model="shadowlilac/aesthetic-shadow-v2", device=device)
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

def write_caption_to_file(file_name, text):
    caption_file = open(file_name, "r")
    lines = caption_file.readlines()
    caption_file.close()
    caption_file = open(file_name, "w")
    caption_file.seek(0)
    caption_file.write(text)
    line = remove_old_tag(lines[0])
    caption_file.write(line)
    caption_file.close()
    
def write_caption(file_name, score):
    if score > 1.50: # out of the scale
        write_caption_to_file(file_name, "out of the scale aesthetic, ")
    if score > 1.10: # out of the scale
        write_caption_to_file(file_name, "masterpiece, ")
    elif score > 0.90:
        write_caption_to_file(file_name, "extremely aesthetic, ")
    elif score > 0.80:
        write_caption_to_file(file_name, "very aesthetic, ")
    elif score > 0.70:
        write_caption_to_file(file_name, "aesthetic, ")
    elif score > 0.50:
        write_caption_to_file(file_name, "slightly aesthetic, ")
    elif score > 0.40:
        write_caption_to_file(file_name, "not displeasing, ")
    elif score > 0.30:
        write_caption_to_file(file_name, "not aesthetic, ")
    elif score > 0.20:
        write_caption_to_file(file_name, "slightly displeasing, ")
    elif score > 0.10:
        write_caption_to_file(file_name, "displeasing, ")
    else:
        write_caption_to_file(file_name, "very displeasing, ")

print("Searching for JPG files...")
file_list = glob.glob('./**/*.jpg')

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

for image in tqdm(file_list):
    try:
        with torch.inference_mode():
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
        getattr(torch, torch.device(device).type).synchronize()
        getattr(torch, torch.device(device).type).empty_cache()
        gc.collect()
        steps_after_gc = 0


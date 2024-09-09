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
pipe.model.eval()
pipe.model.requires_grad_(False)

steps_after_gc = 0


def write_caption_to_file(file_name, text):
    caption_file = open(file_name, "r+")
    lines = caption_file.readlines()
    caption_file.seek(0)
    caption_file.write(text)
    for line in lines:
        caption_file.write(line) 
    caption_file.close()
    
def move_image(image, score):
    if score < 0.90:
        os.makedirs(os.path.dirname(f"bad/{image[2:]}"), exist_ok=True)
        shutil.move(image, f"bad/{image[2:]}")
        try:
            shutil.move(f"{image[:-3]}txt", f"bad/{image[2:-3]}txt")
        except Exception:
            pass
        try:
            shutil.move(f"{image[:-3]}npz", f"bad/{image[2:-3]}npz")
        except Exception:
            pass

print("Searching for JPG files...")
file_list = glob.glob('./*.jpg')

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
os.makedirs(os.path.dirname("bad/"), exist_ok=True)
open("errors/errors.txt", 'a').close()

for image in tqdm(file_list):
    try:
        prediction_single = pipe(images=[image])[0]
        if prediction_single[0]["label"] == "hq":
            prediction = prediction_single[0]["score"]
        elif prediction_single[1]["label"] == "hq":
            prediction = prediction_single[1]["score"]
        move_image(image, prediction)
    except Exception as e:
        print(f"ERROR: {image} MESSAGE: {e}")
        write_caption_to_file("errors/errors.txt", f"\nERROR: {image} MESSAGE: {e}")
        os.makedirs(os.path.dirname(f"errors/{image[2:]}"), exist_ok=True)
        shutil.move(image, f"errors/{image[2:]}")
        try:
            shutil.move(f"{image[:-3]}txt", f"errors/{image[2:-3]}txt")
        except Exception:
            pass
        try:
            shutil.move(f"{image[:-3]}npz", f"errors/{image[2:-3]}npz")
        except Exception:
            pass

    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        getattr(torch, torch.device(device).type).synchronize()
        getattr(torch, torch.device(device).type).empty_cache()
        gc.collect()
        steps_after_gc = 0


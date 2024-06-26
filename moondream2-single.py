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

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
dtype = torch.bfloat16 if device == "xpu" else torch.float16
steps_after_gc = 0

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision, torch_dtype=dtype
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

def write_caption_to_file(file_name, text, line):
    if "english text" in line:
        text_says = line.rsplit(", ")[-1]
        if text[-1] == "\n" or text[-1] == "." or text[-1] == "," or text[-1] == " ":
            text = text[:-1]
        if text[-1] == ".":
            text = text[:-1]
        text = text + ". " + text_says
    caption_file = open(file_name, "w")
    caption_file.seek(0)
    caption_file.write(text)

print("Searching for JPG files...")
file_list = glob.glob('./*.jpg')

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

for image in tqdm(file_list):
    try:
        caption_file = open(image[:-3]+"txt", "r")
        line = caption_file.readlines()[0].replace("\n", "")
        caption_file.close()

        prompt = f"Describe the image in detail. You can use these tags for guidance: {line}\nDon't mention anime, cartoon or illustration and pretend that anime and cartoon is the reality."
        open_image = Image.open(image)
        enc_image = model.encode_image(open_image)
        prediction = model.answer_question(enc_image, prompt, tokenizer).lower()
        open_image.close()

        prediction = prediction.replace("from anime", "").replace("anime-style", "").replace("an anime style", "a").replace("anime style", "").replace("an anime image", "an image").replace("an anime", "a").replace("anime", "")
        prediction = prediction.replace("an animated character", "a character").replace("animated", "").replace("manga girl", "girl").replace("manga male", "male"). replace("manga character","character")
        prediction = prediction.replace("cartoon-style", "").replace("cartoon style", "").replace("a cartoon illustration", "an illustration").replace("a cartoon", "a").replace("cartoon", "")
        prediction = prediction.replace(" an  girl ", " a girl ").replace("\n", " ").replace("  ", " ")
        while prediction[0] == " ":
            prediction = prediction[1:]
        while prediction[-1] == " ":
            prediction = prediction[:-1]
        while prediction[-1] == ".":
            prediction = prediction[:-1]
        if prediction[:9] == "an  girl ":
            prediction = "a girl " + prediction[9:]
        write_caption_to_file(image[:-3]+"txt", prediction, line)
    except Exception as e:
        print(f"ERROR: {image} MESSAGE: {e}")
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

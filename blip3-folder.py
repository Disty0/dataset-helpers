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
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
dtype = torch.bfloat16 if device == "xpu" else torch.float16
steps_after_gc = 0

resolution = 384

model_name = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        return self.eos_sequence in last_ids

def eval_blip3_model(image_path, line):
    global resolution, model, tokenizer, image_processor

    prompt = (
        "<|system|>\nYou are an image tagger. Provide image tags only separated by commas. Don't mention anime, cartoon or illustration and pretend that anime and cartoon is the reality.<|end|>\n"
        f"<|user|>\n<image>\nProvide the most detailed caption. You can use these tags for guidance: {line}\n<|end|>\n<|assistant|>\n"
    )

    image = Image.open(image_path)
    input_image = image.convert("RGB")
    image.close()

    W, H = input_image.size
    aspect_ratio = round(W / H, 2)
    if W < H:
        W = resolution
        H = int(resolution / aspect_ratio)
    elif H < W:
        H = resolution
        W = int(resolution * aspect_ratio)
    if W == H:
        W = resolution
        H = resolution
    image = input_image.resize((W, H), resample=Image.LANCZOS)


    inputs = image_processor([image], return_tensors="pt", image_aspect_ratio="anyres").to(device,dtype=dtype)
    language_inputs = tokenizer([prompt], return_tensors="pt")
    inputs.update(language_inputs)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    generated_text = model.generate(
        **inputs,
        image_size=[image.size],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        max_new_tokens=512,
        top_p=0.95,
        top_k=50,
        num_beams=1,
        stopping_criteria=[EosListStoppingCriteria()],
    )
    prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split(
        "<|end|>"
    )[0]

    return prediction

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
file_list = glob.glob('./**/*.jpg')

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

for image in tqdm(file_list):
    try:
        caption_file = open(image[:-3]+"txt", "r")
        line = caption_file.readlines()[0].replace("\n", "")
        caption_file.close()
        prediction = eval_blip3_model(image, line).lower()
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

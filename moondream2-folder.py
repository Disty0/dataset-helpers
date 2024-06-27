#!/usr/bin/env python

import gc
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
dtype = torch.float16
steps_after_gc = 0
batch_size = 16

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=dtype, attn_implementation="flash_attention_2"
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
file_list = glob.glob('./**/*.jpg')

file_list_batched = []
file_list_temp = []
image_count = len(file_list)
for i in range(image_count):
    file_list_temp.append(file_list[i])
    if len(file_list_temp) >= batch_size or i == image_count-1:
        file_list_batched.append(file_list_temp.copy())
        file_list_temp = []

for image in tqdm(file_list_batched):
    try:
        current_batch_size = range(len(image))
        lines = []
        prompts = []
        open_images = []
        for i in current_batch_size:
            try:
                caption_file = open(image[i][:-3]+"txt", "r")
                line = caption_file.readlines()[0].replace("\n", "")
                lines.append(line)
                caption_file.close()

                prompts.append(f"Describe the image in detail. You can use these tags for guidance: {line}\nDon't mention anime, cartoon or illustration and pretend that anime and cartoon is the reality.")
                open_images.append(Image.open(image[i]).convert("RGB"))
            except Exception as e:
                print(f"ERROR: {image[i]} MESSAGE: {e}")

        prediction = model.batch_answer(images=open_images, prompts=prompts, tokenizer=tokenizer, max_new_tokens=154, repetition_penalty=1.2, do_sample=True)

        for i in current_batch_size:
            try:
                open_images[i].close()
            except Exception as e:
                print(f"ERROR: {image[i]} MESSAGE: {e}")
        for i in current_batch_size:
            try:
                prediction[i] = prediction[i].lower().replace("from anime", "").replace("anime-style", "").replace("an anime style", "a").replace("anime style", "").replace("an anime image", "an image").replace("an anime", "a").replace("anime", "")
                prediction[i] = prediction[i].replace("an animated character", "a character").replace("animated", "").replace("manga girl", "girl").replace("manga male", "male"). replace("manga character","character")
                prediction[i] = prediction[i].replace("cartoon-style", "").replace("cartoon style", "").replace("a cartoon illustration", "an illustration").replace("a cartoon", "a").replace("cartoon", "")
                prediction[i] = prediction[i].replace(" an character", " a character").replace(" an  girl ", " a girl ").replace("\n", " ").replace("  ", " ")
                while prediction[i][0] == " ":
                    prediction[i] = prediction[i][1:]
                while prediction[i][-1] == " ":
                    prediction[i] = prediction[i][:-1]
                while prediction[i][-1] == ".":
                    prediction[i] = prediction[i][:-1]
                if prediction[i][:9] == "an  girl ":
                    prediction[i] = "a girl " + prediction[i][9:]
            except Exception as e:
                print(f"ERROR: {image[i]} MESSAGE: {e}")
        for i in current_batch_size:
            write_caption_to_file(image[i][:-3]+"txt", prediction[i], lines[i])
    except Exception as e:
        print(f"ERROR: {image} MESSAGE: {e}")
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        getattr(torch, torch.device(device).type).synchronize()
        getattr(torch, torch.device(device).type).empty_cache()
        gc.collect()
        steps_after_gc = 0

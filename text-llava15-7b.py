#!/usr/bin/env python

import gc
import torch
try:
    import intel_extension_for_pytorch as ipex
    from ipex_llm import optimize_model
except Exception:
    pass
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, LlavaForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
steps_after_gc = 0

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", low_cpu_mem_usage=True, torch_dtype=torch.float16)
prompt = "<image>\nuser: what does the text say?\nassistant:"

if "xpu" in device:
    model = optimize_model(model).to(device)
else:
    model = model.to(device, dtype=torch.float16)

def write_caption_to_file(file_name, text):
    caption_file = open(file_name, "r")
    line = caption_file.readlines()[0].replace("\n","").removesuffix(",")
    caption_file.close()
    caption_file = open(file_name, "w")
    caption_file.seek(0)
    caption_file.write(line + ", " + text)
    caption_file.close()
    

print("Searching for JPG files...")
file_list_file = open("/mnt/DataSSD/AI/anime_image_dataset/metadata/out/english_text.txt", "r")
file_list = []
for line in file_list_file.readlines():
    file_list.append(f"/mnt/DataSSD/AI/anime_image_dataset/dataset/{line.rsplit('  :::  ')[0][2:-4]}.jpg")
file_list_file.close()


for image_file in tqdm(file_list):
    try:
        image = Image.open(image_file)
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        generate_ids = model.generate(**inputs, max_length=64)
        image.close()
        
        out_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        out_text = out_text.removeprefix("\nuser: what does the text say?\nassistant:").removeprefix(" ").replace("The text says,","The text says")
        out_text = out_text.removesuffix("\n").removesuffix(".").removesuffix('"').replace(",",".").replace("\n"," ") + '"'
        
        write_caption_to_file(image_file[:-3]+"txt", out_text)
    except Exception as e:
        print(f"ERROR: {image_file} MESSAGE: {e}")
    steps_after_gc = steps_after_gc + 1
    if steps_after_gc >= 10000:
        getattr(torch, torch.device(device).type).synchronize()
        getattr(torch, torch.device(device).type).empty_cache()
        gc.collect()
        steps_after_gc = 0


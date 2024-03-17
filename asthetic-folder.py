#!/usr/bin/env python

import os
import gc
import shutil
import glob
import torch

from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image
from tqdm import tqdm


device = "cuda"
aesthetic_path = '/mnt/DataSSD/AI/models/aes-B32-v0.pth'
clip_name = 'openai/clip-vit-base-patch32'
steps_after_gc = 0

def remove_old_tag(text):
    text = text.removeprefix("out of the scale quality, ")
    text = text.removeprefix("masterpiece, ")
    text = text.removeprefix("best quality, ")
    text = text.removeprefix("high quality, ")
    text = text.removeprefix("great quality, ")
    text = text.removeprefix("medium quality, ")
    text = text.removeprefix("normal quality, ")
    text = text.removeprefix("bad quality, ")
    text = text.removeprefix("low quality, ")
    text = text.removeprefix("worst quality, ")
    return text

def image_embeddings(path, model, processor):
    image = Image.open(path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')['pixel_values']
    inputs = inputs.to(device)
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    image.close()
    return (result / np.linalg.norm(result)).squeeze(axis=0)


# binary classifier that consumes CLIP embeddings
class Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = torch.nn.Linear(hidden_size//2, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def write_caption_to_file(file_name, text):
    caption_file = open(file_name, "r+")
    lines = caption_file.readlines()
    caption_file.seek(0)
    caption_file.write(text)
    for line in lines:
        line = remove_old_tag(line)
        caption_file.write(line)
    caption_file.close()
    
def write_caption(file_name, score):
    if score > 1.50: # way out of the scale
        write_caption_to_file(file_name, "out of the scale quality, ")
    elif score > 1.10: # out of the scale
        write_caption_to_file(file_name, "masterpiece, ")
    elif score > 0.98:
        write_caption_to_file(file_name, "best quality, ")
    elif score > 0.90:
        write_caption_to_file(file_name, "high quality, ")
    elif score > 0.75:
        write_caption_to_file(file_name, "great quality, ")
    elif score > 0.50:
        write_caption_to_file(file_name, "medium quality, ")
    elif score > 0.25:
        write_caption_to_file(file_name, "normal quality, ")
    elif score > 0.125:
        write_caption_to_file(file_name, "bad quality, ")
    elif score > 0.025:
        write_caption_to_file(file_name, "low quality, ")
    else:
        write_caption_to_file(file_name, "worst quality, ")

clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name).to(device).eval()

aes_model = Classifier(512, 256, 1).to("cpu")
aes_model.load_state_dict(torch.load(aesthetic_path, map_location="cpu"))
aes_model = aes_model.eval().to(device)

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
open("errors/errors.txt", 'a').close()

print("Searching for JPG files...")

for image in tqdm(glob.glob('./**/*.jpg')):
    try:
        image_embeds = image_embeddings(image, clipmodel, clipprocessor)
        prediction = aes_model(torch.from_numpy(image_embeds).float().to(device))
        write_caption(image[:-3]+"txt", prediction.item())
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


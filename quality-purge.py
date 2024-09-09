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

from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
aesthetic_path = '/mnt/DataSSD/AI/models/aes-B32-v0.pth'
clip_name = 'openai/clip-vit-base-patch32'
steps_after_gc = 0


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
        caption_file.write(line) 
    caption_file.close()
    
def move_image(image, score):
    if score < 0.98:
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


clipprocessor = CLIPProcessor.from_pretrained(clip_name)
clipmodel = CLIPModel.from_pretrained(clip_name).to(device).eval()
clipmodel.requires_grad_(False)
if "xpu" in device:
    clipmodel = ipex.optimize(clipmodel, dtype=torch.float32, inplace=True, weights_prepack=False)
else:
    clipmodel = torch.compile(clipmodel, mode="max-autotune", backend="inductor")

aes_model = Classifier(512, 256, 1).to("cpu")
aes_model.load_state_dict(torch.load(aesthetic_path, map_location="cpu"))
aes_model = aes_model.eval().to(device)
aes_model.requires_grad_(False)
if "xpu" in device:
    aes_model = ipex.optimize(aes_model, dtype=torch.float32, inplace=True, weights_prepack=False)
else:
    aes_model = torch.compile(aes_model, mode="max-autotune", backend="inductor")

print("Searching for JPG files...")
file_list = glob.glob('./*.jpg')

os.makedirs(os.path.dirname("errors/errors.txt"), exist_ok=True)
os.makedirs(os.path.dirname("bad/"), exist_ok=True)
open("errors/errors.txt", 'a').close()

for image in tqdm(file_list):
    try:
        image_embeds = image_embeddings(image, clipmodel, clipprocessor)
        prediction = aes_model(torch.from_numpy(image_embeds).float().to(device))
        move_image(image, prediction.item())
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


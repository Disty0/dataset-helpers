#!/usr/bin/env python

import os
import gc
import glob
import json
import time
import torch
try:
    import intel_extension_for_pytorch as ipex
except Exception:
    pass
from PIL import Image
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
pipe = pipeline("image-classification", model="shadowlilac/aesthetic-shadow-v2", device=device)
pipe.model.eval()
pipe.model.requires_grad_(False)

if "xpu" in device:
    pipe.model = ipex.optimize(pipe.model, inplace=True, weights_prepack=False)

steps_after_gc = 0


class ImageBackend():
    def __init__(self, batches, load_queue_lenght=32, max_load_workers=4):
        self.keep_loading = True
        self.batches = Queue()
        for batch in batches:
            if isinstance(batch, str):
                batch = [batch]
            self.batches.put(batch)
        self.load_queue_lenght = load_queue_lenght
        self.load_queue = Queue()
        self.load_thread = ThreadPoolExecutor(max_workers=max_load_workers)
        for _ in range(max_load_workers):
            self.load_thread.submit(self.load_thread_func)


    def get_images(self):
        return self.load_queue.get()


    def load_thread_func(self):
        while self.keep_loading:
            if self.load_queue.qsize() >= self.load_queue_lenght:
                time.sleep(0.1)
            elif not self.batches.empty():
                batches = self.batches.get()
                curren_batch = []
                for batch in batches:
                    curren_batch.append(self.load_from_file(batch))
                self.load_queue.put(curren_batch)
        print("Stopping the image loader threads")
        return


    def load_from_file(self, image_path):
        image = Image.open(image_path).convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        return [image, image_path]


class SaveAestheticBackend():
    def __init__(self, max_save_workers=2):
        self.keep_saving = True
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)


    def save(self, data, path):
        self.save_queue.put([data,path])


    def save_thread_func(self):
        while self.keep_saving:
            if not self.save_queue.empty():
                data = self.save_queue.get()
                self.save_to_file(data[0], data[1])
            else:
                time.sleep(0.1)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        with open(path, "r") as f:
            json_data = json.load(f)
        json_data["aesthetic-shadow-v2"] = data
        with open(path, "w") as f:
            json.dump(json_data, f)


print("Searching for JSON files...")
file_list = glob.glob('./**/*.json')

image_paths = []

for json_path in tqdm(file_list):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    if json_data.get("aesthetic-shadow-v2", None) is None:
        image_paths.append(os.path.splitext(json_path)[0]+".webp")

epoch_len = len(image_paths)
image_backend = ImageBackend(image_paths)
save_backend = SaveAestheticBackend()

with torch.no_grad():
    for _ in tqdm(range(epoch_len)):
        try:
            image, image_path = image_backend.get_images()[0]
            model_out = pipe(images=[image])[0]
            if model_out[0]["label"] == "hq":
                prediction = model_out[0]["score"]
            elif model_out[1]["label"] == "hq":
                prediction = model_out[1]["score"]
            save_backend.save(prediction, os.path.splitext(image_path)[0]+".json")
        except Exception as e:
            os.makedirs("errors", exist_ok=True)
            error_file = open("errors/errors_aesthetic.txt", 'a')
            error_file.write(f"ERROR: {json_path} MESSAGE: {e} \n")
            error_file.close()
        steps_after_gc = steps_after_gc + 1
        if steps_after_gc >= 10000:
            getattr(torch, torch.device(device).type).synchronize()
            getattr(torch, torch.device(device).type).empty_cache()
            gc.collect()
            steps_after_gc = 0

image_backend.keep_loading = False
image_backend.load_thread.shutdown(wait=True)
del image_backend

while not save_backend.save_queue.empty():
    print(f"Waiting for the remaining writes: {save_backend.save_queue.qsize()}")
    time.sleep(1)
save_backend.keep_saving = False
save_backend.save_thread.shutdown(wait=True)
del save_backend
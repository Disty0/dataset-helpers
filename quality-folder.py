#!/usr/bin/env python

import os
import gc
import glob
import time
import atexit
import torch
try:
    import intel_extension_for_pytorch as ipex
except Exception:
    pass
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

batch_size = 1 # more batch size reduces the accuracy a lot
image_ext = ".webp"
device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
aesthetic_path = '/mnt/DataSSD/AI/models/aes-B32-v0.pth'
clip_name = 'openai/clip-vit-base-patch32'
dtype = torch.float32 # outputs nonsense with 16 bit
steps_after_gc = -1


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

   
def get_quality_tag(score):
    if score > 1.50: # way out of the scale
        return "out of the scale quality"
    elif score > 1.10: # out of the scale
        return "masterpiece"
    elif score > 0.98:
        return "best quality"
    elif score > 0.90:
        return "high quality"
    elif score > 0.75:
        return "great quality"
    elif score > 0.50:
        return "medium quality"
    elif score > 0.25:
        return "normal quality"
    elif score > 0.125:
        return "bad quality"
    elif score > 0.025:
        return "low quality"
    else:
        return "worst quality"


class ImageBackend():
    def __init__(self, batches, processor, load_queue_lenght=256, max_load_workers=4):
        self.load_queue_lenght = 0
        self.keep_loading = True
        self.batches = Queue()
        self.processor = processor
        for batch in batches:
            if isinstance(batch, str):
                batch = [batch]
            self.batches.put(batch)
        self.max_load_queue_lenght = load_queue_lenght
        self.load_queue = Queue()
        self.load_thread = ThreadPoolExecutor()
        for _ in range(max_load_workers):
            self.load_thread.submit(self.load_thread_func)


    def get_images(self):
        result = self.load_queue.get()
        self.load_queue_lenght -= 1
        return result


    def load_thread_func(self):
        while self.keep_loading:
            if self.load_queue_lenght >= self.max_load_queue_lenght:
                time.sleep(0.1)
            elif not self.batches.empty():
                batches = self.batches.get()
                images = []
                image_paths = []
                for batch in batches:
                    image, image_path = self.load_from_file(batch)
                    images.append(image)
                    image_paths.append(image_path)
                inputs = clipprocessor(images=images, return_tensors='pt')['pixel_values'].to(dtype=dtype, memory_format=torch.channels_last)
                self.load_queue.put([inputs, image_paths])
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")
        return


    def load_from_file(self, image_path):
        image = Image.open(image_path).convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        return [image, image_path]


class SaveQualityBackend():
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
                predictions, image_paths = self.save_queue.get()
                for i in range(len(image_paths)):
                    self.save_to_file(predictions[i].item(), os.path.splitext(image_paths[i])[0]+".txt")
            else:
                time.sleep(0.1)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        caption_file = open(path, "r")
        line = caption_file.readlines()[0]
        line = remove_old_tag(line)
        caption_file.close()
        caption_file = open(path, "w")
        caption_file.seek(0)
        caption_file.write(get_quality_tag(data) + ", ")
        caption_file.write(line)
        caption_file.close()


if __name__ == '__main__':
    clipprocessor = CLIPProcessor.from_pretrained(clip_name)
    clipmodel = CLIPModel.from_pretrained(clip_name).eval().to(device, dtype=dtype, memory_format=torch.channels_last)
    clipmodel.requires_grad_(False)
    if "xpu" in device:
        clipmodel = ipex.optimize(clipmodel, dtype=dtype, inplace=True, weights_prepack=False)
    else:
        clipmodel = torch.compile(clipmodel, mode="max-autotune", backend="inductor")

    aes_model = Classifier(512, 256, 1).to("cpu")
    aes_model.load_state_dict(torch.load(aesthetic_path, map_location="cpu"))
    aes_model = aes_model.eval().to(device, dtype=dtype, memory_format=torch.channels_last)
    aes_model.requires_grad_(False)
    if "xpu" in device:
        aes_model = ipex.optimize(aes_model, dtype=dtype, inplace=True, weights_prepack=False)
    else:
        aes_model = torch.compile(aes_model, mode="max-autotune", backend="inductor")


    print(f"Searching for {image_ext} files...")
    file_list = glob.glob(f'./**/*{image_ext}')
    batches = []
    current_batch = []
    for file in file_list:
        current_batch.append(file)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    if len(current_batch) != 0:
        batches.append(current_batch)

    epoch_len = len(batches)
    image_backend = ImageBackend(batches, clipprocessor)
    save_backend = SaveQualityBackend()

    def exit_handler(image_backend, save_backend):
        image_backend.keep_loading = False
        image_backend.load_thread.shutdown(wait=True)
        del image_backend

        while not save_backend.save_queue.empty():
            print(f"Waiting for the remaining writes: {save_backend.save_queue.qsize()}")
            time.sleep(1)
        save_backend.keep_saving = False
        save_backend.save_thread.shutdown(wait=True)
        del save_backend
    atexit.register(exit_handler, image_backend, save_backend)

    with torch.no_grad():
        for _ in tqdm(range(epoch_len)):
            try:
                inputs, image_paths = image_backend.get_images()
                image_embeds = clipmodel.get_image_features(pixel_values=inputs.to(device))
                image_embeds = (image_embeds / torch.linalg.norm(image_embeds))
                predictions = aes_model(image_embeds)
                save_backend.save(predictions, image_paths)
            except Exception as e:
                os.makedirs("errors", exist_ok=True)
                error_file = open("errors/errors.txt", 'a')
                error_file.write(f"ERROR: {image_paths} MESSAGE: {e} \n")
                error_file.close()
            steps_after_gc = steps_after_gc + 1
            if steps_after_gc == 0 or steps_after_gc >= 10000:
                if "cpu" not in device:
                    getattr(torch, torch.device(device).type).synchronize()
                    getattr(torch, torch.device(device).type).empty_cache()
                gc.collect()
                steps_after_gc = 1 if steps_after_gc == 0 else 0

    atexit.unregister(exit_handler)
    exit_handler(image_backend, save_backend)

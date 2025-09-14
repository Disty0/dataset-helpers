#!/usr/bin/env python

import os
import gc
import json
import time
import atexit
import torch

try:
    import intel_extension_for_pytorch as ipex # noqa: F401
except Exception:
    pass

import pytorch_lightning as pl
import huggingface_hub
from transformers import CLIPModel, CLIPProcessor
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tqdm import tqdm

try:
    import pillow_jxl # noqa: F401
except Exception:
    pass
from PIL import Image # noqa: E402

from typing import List, Tuple

batch_size = 32
use_tunable_ops = False
use_torch_compile = True
device = torch.device("xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
caption_key = "waifu-scorer-v3"
model_repo = "Eugeoter/waifu-scorer-v3"
model_filename = "model.pth"
clip_repo = "openai/clip-vit-large-patch14"
dtype = torch.float16 if device.type != "cpu" else torch.float32
img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


class MLP(pl.LightningModule):
    def __init__(self, input_size: int, xcol: str = 'emb', ycol: str = 'avg_rating', batch_norm: bool = True):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 2048),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(2048) if batch_norm else torch.nn.Identity(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512) if batch_norm else torch.nn.Identity(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256) if batch_norm else torch.nn.Identity(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128) if batch_norm else torch.nn.Identity(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x: torch.FloatTensor) -> List[float]:
        out = self.layers(x).clamp(0, 10) / 10
        return out.cpu().numpy().reshape(-1).tolist()


class ImageBackend():
    def __init__(self, batches: List[List[str]], processor: CLIPProcessor, load_queue_lenght: int = 256, max_load_workers: int = 4):
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

    def get_images(self) -> Tuple[torch.FloatTensor, List[str]]:
        result = self.load_queue.get()
        self.load_queue_lenght -= 1
        return result

    @torch.no_grad()
    def load_thread_func(self) -> None:
        while self.keep_loading:
            if self.load_queue_lenght >= self.max_load_queue_lenght:
                time.sleep(0.1)
            elif not self.batches.empty():
                batches = self.batches.get()
                images = []
                for batch in batches:
                    images.append(self.load_from_file(batch))
                inputs = self.processor(images=images, return_tensors='pt')['pixel_values'].to(dtype=dtype)
                self.load_queue.put((inputs, batches))
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")

    def load_from_file(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        return image


class SaveAestheticBackend():
    def __init__(self, max_save_workers: int = 2):
        self.keep_saving = True
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)

    def save(self, data: List[float], path: List[str]) -> None:
        self.save_queue.put((data,path))

    @torch.no_grad()
    def save_thread_func(self) -> None:
        while self.keep_saving:
            if not self.save_queue.empty():
                predictions, image_paths = self.save_queue.get()
                for i in range(len(image_paths)):
                    self.save_to_file(predictions[i], os.path.splitext(image_paths[i])[0]+".json")
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")

    def save_to_file(self, data: float, path: str) -> None:
        with open(path, "r") as f:
            json_data = json.load(f)
        json_data[caption_key] = data
        with open(path, "w") as f:
            json.dump(json_data, f)


@torch.no_grad()
def main():
    steps_after_gc = -1

    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

    if use_tunable_ops:
        torch.cuda.tunable.enable(val=True)

    clipprocessor = CLIPProcessor.from_pretrained(clip_repo, use_fast=True)
    clipmodel = CLIPModel.from_pretrained(clip_repo).eval().to(device, dtype=dtype)
    clipmodel.requires_grad_(False)
    if device.type == "cpu":
        import openvino.properties.hint as ov_hints
        clipmodel.get_image_features = torch.compile(clipmodel.get_image_features, backend="openvino", options={"device": "GPU", "config" : {ov_hints.execution_mode : ov_hints.ExecutionMode.ACCURACY}})
    elif use_torch_compile:
        clipmodel.get_image_features = torch.compile(clipmodel.get_image_features, backend="inductor")

    aes_model = MLP(input_size=768).to("cpu")
    aes_model.load_state_dict(torch.load(
        huggingface_hub.hf_hub_download(
            repo_id=model_repo,
            repo_type='model',
            filename=model_filename
        ),
        map_location="cpu"))
    aes_model = aes_model.eval().to(device, dtype=dtype)
    aes_model.requires_grad_(False)
    if device.type == "cpu":
        aes_model.forward = torch.compile(aes_model.forward, backend="openvino", options={"device": "GPU", "config" : {ov_hints.execution_mode : ov_hints.ExecutionMode.ACCURACY}})
    else:
        aes_model.forward = torch.compile(aes_model.forward, backend="inductor")


    print(f"Searching for {img_ext_list} files...")
    file_list = []
    for ext in img_ext_list:
        file_list.extend(glob(f"**/*.{ext}"))
    image_paths = []

    for image_path in tqdm(file_list):
        try:
            json_path = os.path.splitext(image_path)[0]+".json"
            with open(json_path, "r") as f:
                json_data = json.load(f)
            if json_data.get(caption_key, None) is None:
                image_paths.append(image_path)
        except Exception as e:
            print(f"ERROR: {json_path} MESSAGE: {e}")

    batches = []
    current_batch = []
    for file in image_paths:
        current_batch.append(file)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    if len(current_batch) != 0:
        batches.append(current_batch)

    epoch_len = len(batches)
    image_backend = ImageBackend(batches, clipprocessor)
    save_backend = SaveAestheticBackend()

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

    for _ in tqdm(range(epoch_len)):
        try:
            inputs, image_paths = image_backend.get_images()
            image_embeds = clipmodel.get_image_features(pixel_values=inputs.to(device))
            image_embeds_l2 = image_embeds.norm(2, -1, keepdim=True)
            image_embeds_l2[image_embeds_l2 == 0] = 1
            image_embeds = image_embeds / image_embeds_l2
            predictions = aes_model(image_embeds)
            save_backend.save(predictions, image_paths)
        except Exception as e:
            os.makedirs("errors", exist_ok=True)
            error_file = open("errors/errors.txt", 'a')
            error_file.write(f"ERROR: {image_paths} MESSAGE: {e} \n")
            error_file.close()
        steps_after_gc = steps_after_gc + 1
        if steps_after_gc == 0 or steps_after_gc >= 10000:
            gc.collect()
            if device.type != "cpu":
                getattr(torch, device.type).synchronize()
                getattr(torch, device.type).empty_cache()
            steps_after_gc = 1 if steps_after_gc == 0 else 0

    atexit.unregister(exit_handler)
    exit_handler(image_backend, save_backend)

if __name__ == '__main__':
    main()

#!/usr/bin/env python

from typing import Optional

import os
import gc
import json
import time
import atexit

import torch
if torch.version.hip:
    try:
        # don't use this for training models, only for inference with latent encoder and embed encoder
        # https://github.com/huggingface/diffusers/discussions/7172
        from functools import wraps
        from flash_attn import flash_attn_func
        sdpa_pre_flash_atten = torch.nn.functional.scaled_dot_product_attention
        @wraps(sdpa_pre_flash_atten)
        def sdpa_flash_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: Optional[torch.FloatTensor] = None, dropout_p: float = 0.0, is_causal: bool = False, scale: Optional[float] = None, enable_gqa: bool = False, **kwargs) -> torch.FloatTensor:
            if query.shape[-1] <= 128 and attn_mask is None and query.dtype != torch.float32:
                is_unsqueezed = False
                if query.dim() == 3:
                    query = query.unsqueeze(0)
                    is_unsqueezed = True
                    if key.dim() == 3:
                        key = key.unsqueeze(0)
                    if value.dim() == 3:
                        value = value.unsqueeze(0)
                if enable_gqa:
                    key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                    value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                attn_output = flash_attn_func(q=query, k=key, v=value, dropout_p=dropout_p, causal=is_causal, softmax_scale=scale).transpose(1, 2)
                if is_unsqueezed:
                    attn_output = attn_output.squeeze(0)
                return attn_output
            else:
                if enable_gqa:
                    kwargs["enable_gqa"] = enable_gqa
                return sdpa_pre_flash_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = sdpa_flash_atten
    except Exception as e:
        print(f"Failed to enable Flash Atten for ROCm: {e}")

from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from transformers import ViTForImageClassification, ViTImageProcessor
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
caption_key = "aesthetic-shadow-v2"
model_repo = "Disty0/aesthetic-shadow-v2"
dtype = torch.float16 if device.type != "cpu" else torch.float32
img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


class ImageBackend():
    def __init__(self, batches: List[List[str]], processor: ViTImageProcessor, load_queue_lenght: int = 256, max_load_workers: int = 4):
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
                inputs = self.processor(images=images, return_tensors="pt")["pixel_values"].to(dtype=dtype)
                self.load_queue.put((inputs, batches))
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")

    def load_from_file(self, image_path: str) -> Image.Image:
        image = Image.open(image_path).convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
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
                    self.save_to_file(predictions[i][0].item(), os.path.splitext(image_paths[i])[0]+".json")
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

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

    if use_tunable_ops:
        torch.cuda.tunable.enable(val=True)

    model = ViTForImageClassification.from_pretrained(model_repo, dtype=dtype)
    processor = ViTImageProcessor.from_pretrained(model_repo, use_fast=True)
    model = model.to(device, dtype=dtype).eval()
    model.requires_grad_(False)

    if device.type == "cpu":
        model = torch.compile(model, backend="openvino", options = {"device" : "GPU"})
    elif use_torch_compile:
        model = torch.compile(model, backend="inductor")

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
    image_backend = ImageBackend(batches, processor)
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
            inputs = inputs.to(device)
            prediction = torch.nn.functional.softmax(model(inputs).logits, dim=-1)
            save_backend.save(prediction, image_paths)
        except Exception as e:
            os.makedirs("errors", exist_ok=True)
            error_file = open("errors/errors_aesthetic.txt", "a")
            error_file.write(f"ERROR: {json_path} MESSAGE: {e} \n")
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

if __name__ == "__main__":
    main()

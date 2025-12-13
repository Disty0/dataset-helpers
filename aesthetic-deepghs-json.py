#!/usr/bin/env python

import os
import gc
import json
import time
import atexit
import huggingface_hub
import numpy as np
import pandas as pd
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tqdm import tqdm
import onnxruntime as ort

try:
    import pillow_jxl # noqa: F401
except Exception:
    pass
from PIL import Image # noqa: E402

from typing import List, Tuple

batch_size = 32
model_repo = "deepghs/anime_aesthetic"
model_name = "swinv2pv3_v0_448_ls0.2_x"
model_filename = "model.onnx"
label_filename = "meta.json"
samples_filename = "samples.csv"
img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


class ImageBackend():
    def __init__(self, batches: List[List[str]], load_queue_lenght: int = 256, max_load_workers: int = 4):
        self.load_queue_lenght = 0
        self.keep_loading = True
        self.batches = Queue()
        for batch in batches:
            if isinstance(batch, str):
                batch = [batch]
            self.batches.put(batch)
        self.max_load_queue_lenght = load_queue_lenght
        self.load_queue = Queue()
        self.load_thread = ThreadPoolExecutor()
        for _ in range(max_load_workers):
            self.load_thread.submit(self.load_thread_func)

    def get_images(self) -> Tuple[np.ndarray, List[str]]:
        result = self.load_queue.get()
        self.load_queue_lenght -= 1
        return result

    def load_thread_func(self) -> None:
        while self.keep_loading:
            if self.load_queue_lenght >= self.max_load_queue_lenght:
                time.sleep(0.25)
            elif not self.batches.empty():
                batches = self.batches.get()
                images = []
                for batch in batches:
                    image = self.load_from_file(batch)
                    images.append(image)
                images = np.array(images).astype(np.float32)
                self.load_queue.put((images, batches))
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")

    def load_from_file(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as img:
            background = Image.new("RGBA", img.size, (255, 255, 255))
            image = Image.alpha_composite(background, img.convert("RGBA")).convert("RGB")
        image = image.resize((448, 448), Image.BICUBIC)
        image_array = np.asarray(image)
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = (image_array / 255.0).astype(np.float32)
        mean = np.asarray([0.5]).reshape((-1, 1, 1))
        std = np.asarray([0.5]).reshape((-1, 1, 1))
        image_array = (image_array - mean) / std
        return image_array


class SaveTagBackend():
    def __init__(self, model_config: dict, mark_table: List[np.ndarray], max_save_workers: int = 2):
        self.model_config = model_config
        self.mark_table = mark_table
        self.keep_saving = True
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)

    def save(self, data: np.ndarray, path: List[str]) -> None:
        self.save_queue.put((data,path))

    def save_thread_func(self) -> None:
        while self.keep_saving:
            if not self.save_queue.empty():
                predictions, image_paths = self.save_queue.get()
                for i in range(len(image_paths)):
                    self.save_to_file(self.get_tags(predictions[i]), os.path.splitext(image_paths[i])[0]+".json")
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")

    def save_to_file(self, data: Tuple[str, float], path: str) -> None:
        with open(path, "r") as json_file:
            json_data = json.load(json_file)
        json_data[model_name] = data[0]
        json_data[model_name+"_percentile"] = data[1]
        with open(path, "w") as f:
            json.dump(json_data, f)

    def get_tags(self, predictions: np.ndarray) -> Tuple[str, float]:
        values = dict(zip(self.model_config["labels"], map(lambda x: x.item(), predictions)))
        weighted_mean = sum(i * values[label] for i, label in enumerate(self.model_config["labels"]))
        idx = np.searchsorted(self.mark_table[0], np.clip(weighted_mean, a_min=0.0, a_max=6.0))
        if idx < self.mark_table[0].shape[0] - 1:
            x0, y0 = self.mark_table[0][idx], self.mark_table[1][idx]
            x1, y1 = self.mark_table[0][idx + 1], self.mark_table[1][idx + 1]
            percentile = np.clip((weighted_mean - x0) / (x1 - x0) * (y1 - y0) + y0, a_min=0.0, a_max=1.0)

        else:
            percentile = self.mark_table[1][idx]
        return (max(values, key=values.get), 1-percentile)


def main():
    steps_after_gc = -1
    model_config_path = huggingface_hub.hf_hub_download(
        repo_id=model_repo,
        repo_type="model",
        filename=model_name + "/" + label_filename,
    )
    model_path = huggingface_hub.hf_hub_download(
        repo_id=model_repo,
        repo_type="model",
        filename=model_name + "/" + model_filename,
    )

    with open(model_config_path, "r") as model_config_file:
        model_config = json.load(model_config_file)

    df = pd.read_csv(huggingface_hub.hf_hub_download(
        repo_id=model_repo,
        repo_type="model",
        filename=model_name + "/" + samples_filename,
    ))
    df = df.sort_values(["score"])
    df["cnt"] = list(range(len(df)))
    df["final_score"] = df["cnt"] / len(df)

    mark_table = [np.concatenate([[0.0], df["score"], [6.0]]), np.concatenate([[0.0], df["final_score"], [1.0]])]
    del df

    if "OpenVINOExecutionProvider" in ort.get_available_providers():
        # requires provider options for gpu support
        model = ort.InferenceSession(
            model_path,
            providers=(["OpenVINOExecutionProvider"]),
            provider_options=[{"device_type" : "GPU", "precision": "FP16"}],
        )
    else:
        model = ort.InferenceSession(
            model_path,
            providers=(
                ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else
                ["ROCMExecutionProvider"] if "ROCMExecutionProvider" in ort.get_available_providers() else
                ["CPUExecutionProvider"]
            ),
        )


    print(f"Searching for {img_ext_list} files...")
    file_list = []
    for ext in img_ext_list:
        file_list.extend(glob(f"**/*.{ext}"))
    image_paths = []

    for image_path in tqdm(file_list):
        try:
            json_path = os.path.splitext(image_path)[0]+".json"
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)
            if json_data.get(model_name, None) is None:
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
    image_backend = ImageBackend(batches)
    save_backend = SaveTagBackend(model_config, mark_table)

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
            images, image_paths = image_backend.get_images()
            predictions = model.run(["output"], {"input": images})[0]
            save_backend.save(predictions, image_paths)
        except Exception as e:
            os.makedirs("errors", exist_ok=True)
            error_file = open("errors/errors.txt", "a")
            error_file.write(f"ERROR: {image_paths} MESSAGE: {e} \n")
            error_file.close()
        steps_after_gc = steps_after_gc + 1
        if steps_after_gc == 0 or steps_after_gc >= 10000:
            gc.collect()
            steps_after_gc = 1 if steps_after_gc == 0 else 0

    atexit.unregister(exit_handler)
    exit_handler(image_backend, save_backend)

if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import gc
import glob
import json
import time
import atexit
import huggingface_hub
import numpy as np
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import onnxruntime as ort

batch_size = 32
image_ext = ".jxl"
model_repo = "deepghs/anime_style_ages"
MODEL_FILENAME = "mobilenetv3_v0_dist/model.onnx"
LABEL_FILENAME = "mobilenetv3_v0_dist/meta.json"
steps_after_gc = -1

if image_ext == ".jxl":
    import pillow_jxl # noqa: F401
from PIL import Image # noqa: E402
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


class ImageBackend():
    def __init__(self, batches, load_queue_lenght=256, max_load_workers=4):
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


    def get_images(self):
        result = self.load_queue.get()
        self.load_queue_lenght -= 1
        return result


    def load_thread_func(self):
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
                self.load_queue.put([images, batches])
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")
        return


    def load_from_file(self, image_path):
        image = Image.open(image_path).convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        image = image.resize((384, 384), Image.BICUBIC)
        image_array = np.asarray(image)
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = (image_array / 255.0).astype(np.float32)
        mean = np.asarray([0.5]).reshape((-1, 1, 1))
        std = np.asarray([0.5]).reshape((-1, 1, 1))
        image_array = (image_array - mean) / std
        return image_array


class SaveQualityBackend():
    def __init__(self, model_config, max_save_workers=2):
        self.model_config = model_config
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
                    self.save_to_file(self.get_tags(predictions[i]), os.path.splitext(image_paths[i])[0]+".json")
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        with open(path, "r") as json_file:
            json_data = json.load(json_file)
        json_data["style_age"] = data
        if json_data.get("created_at", "none") == "none":
            json_data["created_at"] = data
        with open(path, "w") as f:
            json.dump(json_data, f)


    def get_tags(self, predictions):
        values = dict(zip(self.model_config["labels"], map(lambda x: x.item(), predictions)))
        return max(values, key=values.get).replace("-", "")


if __name__ == '__main__':
    model_config_path = huggingface_hub.hf_hub_download(
        model_repo,
        LABEL_FILENAME,
    )
    model_path = huggingface_hub.hf_hub_download(
        model_repo,
        MODEL_FILENAME,
    )

    with open(model_config_path, "r") as model_config_file:
        model_config = json.load(model_config_file)

    if "OpenVINOExecutionProvider" in ort.get_available_providers():
        # requires provider options for gpu support
        model = ort.InferenceSession(
            model_path,
            providers=(["OpenVINOExecutionProvider"]),
            provider_options=[{'device_type' : "GPU", "precision": "FP16"}],
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


    print(f"Searching for {image_ext} files...")
    file_list = glob.glob(f'**/*{image_ext}')
    image_paths = []

    for image_path in tqdm(file_list):
        try:
            json_path = os.path.splitext(image_path)[0]+".json"
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)
            if not json_data.get("style_age", ""):
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
    save_backend = SaveQualityBackend(model_config)

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
            predictions = model.run(['output'], {'input': images})[0]
            save_backend.save(predictions, image_paths)
        except Exception as e:
            os.makedirs("errors", exist_ok=True)
            error_file = open("errors/errors.txt", 'a')
            error_file.write(f"ERROR: {image_paths} MESSAGE: {e} \n")
            error_file.close()
        steps_after_gc = steps_after_gc + 1
        if steps_after_gc == 0 or steps_after_gc >= 10000:
            gc.collect()
            steps_after_gc = 1 if steps_after_gc == 0 else 0

    atexit.unregister(exit_handler)
    exit_handler(image_backend, save_backend)

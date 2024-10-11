#!/usr/bin/env python

import os
import gc
import glob
import json
import time
import atexit
import huggingface_hub
import numpy as np
import pandas as pd
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import onnxruntime as ort

batch_size = 24
image_ext = ".jxl"
general_thresh = 0.35
character_thresh = 0.5
model_repo = "SmilingWolf/wd-swinv2-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
steps_after_gc = -1

if image_ext == ".jxl":
    import pillow_jxl # noqa: F401
from PIL import Image # noqa: E402


rating_map = {
    "general": "g",
    "sensitive": "s",
    "questionable": "q",
    "explicit": "e",
}


class ImageBackend():
    def __init__(self, batches, model_target_size, load_queue_lenght=256, max_load_workers=4):
        self.load_queue_lenght = 0
        self.keep_loading = True
        self.batches = Queue()
        self.model_target_size = model_target_size
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
                for batch in batches:
                    image = self.load_from_file(batch)
                    images.append(image)
                images = np.array(images)
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
        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != self.model_target_size:
            padded_image = padded_image.resize(
                (self.model_target_size, self.model_target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return image_array


class SaveQualityBackend():
    def __init__(self, tag_names, rating_indexes, character_indexes, general_indexes, max_save_workers=2):
        self.tag_names = tag_names
        self.rating_indexes = rating_indexes
        self.character_indexes = character_indexes
        self.general_indexes = general_indexes
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
                time.sleep(0.1)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        rating, character_res, sorted_general_strings = data[0], data[1], data[2]
        json_data = {}
        json_data["rating"] = rating
        json_data["tag_string_character"] = character_res
        json_data["tag_string_general"] = sorted_general_strings
        #json_data["special_tags"] = "visual_novel_cg"
        #txt_path = os.path.splitext(path)[0]+".txt"
        #with open(txt_path, "r") as txt_file:
        #    copyright_tag = txt_file.readlines()[0].split(", ", maxsplit=4)[3]
        #    if copyright_tag.startswith("from "):
        #        json_data["tag_string_copyright"] = copyright_tag.removeprefix("from ").replace(" ", "_")
        if not json_data.get("tag_string_copyright", ""):
            json_data["tag_string_copyright"] = ""
        if not json_data.get("tag_string_artist", ""):
            json_data["tag_string_artist"] = ""
        if not json_data.get("tag_string_meta", ""):
            json_data["tag_string_meta"] = ""
        if not json_data.get("created_at", ""):
            json_data["created_at"] = "none"
        with open(path, "w") as f:
            json.dump(json_data, f)


    def get_tags(self, predictions):
        labels = list(zip(self.tag_names, predictions.astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)
        rating = max(rating, key=rating.get)
        rating = rating_map[rating]

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = " ".join(sorted_general_strings)

        return [rating, character_res, sorted_general_strings]


if __name__ == '__main__':
    csv_path = huggingface_hub.hf_hub_download(
        model_repo,
        LABEL_FILENAME,
    )
    model_path = huggingface_hub.hf_hub_download(
        model_repo,
        MODEL_FILENAME,
    )

    dataframe = pd.read_csv(csv_path)
    name_series = dataframe["name"]
    tag_names = name_series.tolist()
    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])

    if "OpenVINOExecutionProvider" in ort.get_available_providers():
        # requires provider options for gpu support
        # fp16 causes nonsense outputs
        model = ort.InferenceSession(
            model_path,
            providers=(["OpenVINOExecutionProvider"]),
            provider_options=[{'device_type' : "GPU", "precision": "FP32"}],
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
    _, height, width, _ = model.get_inputs()[0].shape
    model_target_size = height

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name


    print(f"Searching for {image_ext} files...")
    file_list = glob.glob(f'**/*{image_ext}')
    image_paths = []

    for image_path in tqdm(file_list):
        try:
            json_path = os.path.splitext(image_path)[0]+".json"
            if not os.path.exists(json_path):
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
    image_backend = ImageBackend(batches, model_target_size)
    save_backend = SaveQualityBackend(tag_names, rating_indexes, character_indexes, general_indexes)

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
        #try:
        images, image_paths = image_backend.get_images()
        predictions = model.run([label_name], {input_name: images})[0]
        save_backend.save(predictions, image_paths)
        #except Exception as e:
        #    os.makedirs("errors", exist_ok=True)
        #    error_file = open("errors/errors.txt", 'a')
        #    error_file.write(f"ERROR: {image_paths} MESSAGE: {e} \n")
        #    error_file.close()
        steps_after_gc = steps_after_gc + 1
        if steps_after_gc == 0 or steps_after_gc >= 10000:
            gc.collect()
            steps_after_gc = 1 if steps_after_gc == 0 else 0

    atexit.unregister(exit_handler)
    exit_handler(image_backend, save_backend)

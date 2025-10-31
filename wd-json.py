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
from tqdm import tqdm
from glob import glob
import onnxruntime as ort

try:
    import pillow_jxl # noqa: F401
except Exception:
    pass
from PIL import Image # noqa: E402

from typing import List, Tuple

batch_size = 24
general_thresh = 0.35
character_thresh = 0.5

model_repo = "SmilingWolf/wd-swinv2-tagger-v3"
#model_repo = "deepghs/pixai-tagger-v0.9-onnx"

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"
img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970
caption_key = model_repo.rsplit("/", maxsplit=1)[-1].split("-", maxsplit=1)[0].replace(".", "-").lower()


rating_map = {
    "general": "g",
    "sensitive": "s",
    "questionable": "q",
    "explicit": "e",
}


class ImageBackend():
    def __init__(self, batches: List[List[str]], model_target_size: int, channels_first: bool, load_queue_lenght: int = 256, max_load_workers: int = 4):
        self.load_queue_lenght = 0
        self.keep_loading = True
        self.batches = Queue()
        self.model_target_size = (model_target_size, model_target_size)
        self.channels_first = channels_first
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
                images = np.array(images)
                self.load_queue.put((images, batches))
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")

    def load_from_file(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGBA")
        image_size = image.size

        background = Image.new("RGBA", image_size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")

        if image_size != self.model_target_size:
            if self.channels_first:
                image = image.resize(self.model_target_size, Image.BILINEAR)
            else:
                max_dim = max(image_size)
                pad_left = (max_dim - image_size[0]) // 2
                pad_top = (max_dim - image_size[1]) // 2
                padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
                padded_image.paste(image, (pad_left, pad_top))
                image = padded_image.resize(self.model_target_size, Image.BICUBIC)

        image = np.asarray(image, dtype=np.float32)
        if self.channels_first:
            image = (image.transpose(2,0,1) / 127.5) - 1
        else:
            image = image[:, :, ::-1] # RGB to BGR

        return image


class SaveTagBackend():
    def __init__(self, tag_names: List[str], rating_indexes: List[int], character_indexes: List[int], general_indexes: List[int], max_save_workers: int = 2):
        self.tag_names = tag_names
        self.rating_indexes = rating_indexes
        self.character_indexes = character_indexes
        self.general_indexes = general_indexes
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
                    self.save_to_file(self.get_tags(predictions[i]), image_paths[i])
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")

    def save_to_file(self, data: List[str], image_path: str) -> None:
        rating, character_strings, sorted_general_strings = data[0], data[1], data[2]
        base_name, file_ext = os.path.splitext(image_path)
        file_ext = file_ext[1:]
        json_path = base_name+".json"

        if os.path.exists(json_path):
            with open(json_path, "r") as json_file:
                json_data = json.load(json_file)

            if json_data.get("rating", None) is None:
                json_data["rating"] = rating
            if json_data.get("tag_string_character", None) is None:
                json_data["tag_string_character"] = character_strings
            if json_data.get("tag_string_general", None) is None:
                json_data["tag_string_general"] = sorted_general_strings

            if json_data.get("tag_string_copyright", None) is None:
                json_data["tag_string_copyright"] = ""
            if json_data.get("tag_string_artist", None) is None:
                json_data["tag_string_artist"] = ""
            if json_data.get("tag_string_meta", None) is None:
                json_data["tag_string_meta"] = ""

            if json_data.get("created_at", None) is None:
                json_data["created_at"] = "none"
            if json_data.get("source", None) is None:
                json_data["source"] = "wd-json.py"

            if json_data.get("file_ext", None) is None:
                json_data["file_ext"] = file_ext
            if json_data.get("file_size", None) is None:
                json_data["file_size"] = os.path.getsize(image_path)
        else:
            json_data = {}
            json_data["rating"] = rating
            json_data["tag_string_character"] = character_strings
            json_data["tag_string_general"] = sorted_general_strings

            json_data["tag_string_copyright"] = ""
            json_data["tag_string_artist"] = ""
            json_data["tag_string_meta"] = ""

            json_data["created_at"] = "none"
            json_data["source"] = "wd-json.py"

            json_data["file_ext"] = file_ext
            json_data["file_size"] = os.path.getsize(image_path)

        if rating is not None:
            json_data[f"{caption_key}_rating"] = rating
        json_data[f"{caption_key}_tag_string_character"] = character_strings
        json_data[f"{caption_key}_tag_string_general"] = sorted_general_strings
        #json_data["special_tags"] = "visual_novel_cg"

        with open(json_path, "w") as f:
            json.dump(json_data, f)

    def get_tags(self, predictions: np.ndarray) -> Tuple[str, str, str]:
        labels = list(zip(self.tag_names, predictions.astype(float)))

        if len(self.rating_indexes) > 0:
            rating = dict([labels[i] for i in self.rating_indexes])
            rating = rating_map[max(rating, key=rating.get)]
        else:
            rating = None

        character_names = [labels[i] for i in self.character_indexes]
        character_res = dict([x for x in character_names if x[1] > character_thresh and isinstance(x[0], str)])
        character_strings = sorted(character_res.items(), key=lambda x: x[1], reverse=True)
        character_strings = [x[0] for x in character_strings]
        character_strings = " ".join(character_strings)

        general_names = [labels[i] for i in self.general_indexes]
        general_res = dict([x for x in general_names if x[1] > general_thresh and isinstance(x[0], str)])
        general_strings = sorted(general_res.items(), key=lambda x: x[1], reverse=True)
        general_strings = [x[0] for x in general_strings]
        general_strings = " ".join(general_strings)

        return (rating, character_strings, general_strings)


def main():
    steps_after_gc = -1
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
            provider_options=[{"device_type" : "GPU", "precision": "FP32"}],
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

    # wd_in: batch_size, 448, 448, 3
    # wd_out: output
    # pixai_in: batch_size, 3, 448, 448
    # pixai_out embedding, logits, prediction

    input_shapes = model.get_inputs()[-1]
    output_shapes = model.get_outputs()[-1]

    model_target_size = input_shapes.shape[2]
    channels_first = bool(input_shapes.shape[1] == 3)

    input_name = input_shapes.name
    label_name = output_shapes.name


    print(f"Searching for {img_ext_list} files...")
    file_list = []
    for ext in img_ext_list:
        file_list.extend(glob(f"**/*.{ext}"))
    image_paths = []

    for image_path in tqdm(file_list):
        try:
            json_path = os.path.splitext(image_path)[0]+".json"
            if os.path.exists(json_path):
                with open(json_path, "r") as json_file:
                    json_data = json.load(json_file)
                if not json_data.get(f"{caption_key}_tag_string_general", ""):
                    image_paths.append(image_path)
            else:
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
    image_backend = ImageBackend(batches, model_target_size, channels_first)
    save_backend = SaveTagBackend(tag_names, rating_indexes, character_indexes, general_indexes)

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
            predictions = model.run([label_name], {input_name: images})[-1]
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

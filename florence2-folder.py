#!/usr/bin/env python

import os
import gc
import glob
import json
import time
import atexit
import torch
try:
    import transformers # ipex hijacks transformers and makes it unable to load a model
    backup_get_class_from_dynamic_module = transformers.dynamic_module_utils.get_class_from_dynamic_module
    import intel_extension_for_pytorch as ipex
    ipex.llm.utils._get_class_from_dynamic_module = backup_get_class_from_dynamic_module
    transformers.dynamic_module_utils.get_class_from_dynamic_module = backup_get_class_from_dynamic_module
except Exception:
    pass
from PIL import Image
from queue import Queue
from transformers import AutoModelForCausalLM, AutoProcessor
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

batch_size = 8
image_ext = ".webp"
model_id = "MiaoshouAI/Florence-2-base-PromptGen-v1.5"
device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
dtype = torch.float16 if "cuda" in device else torch.bfloat16 if "xpu" in device else torch.float32
use_flash_atten = "cuda" in device
steps_after_gc = -1


if not use_flash_atten:
    import transformers
    from transformers.dynamic_module_utils import get_imports
    def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
        return imports
    transformers.dynamic_module_utils.get_imports = fixed_get_imports


meta_blacklist = [
    "source",
    "upload",
    "annotated",
    "translation",
    "translated",
    "completion_time",
    "_request",
    "_id",
    "_link",
    "_available",
    "_account",
    "_mismatch",
    "_sample",
    "check_",
    "has_",
    "metadata",
    "thumbnail",
    "duplicate",
    "revision",
    "variant_set",
    "commentary",
    "audio",
    "video",
    "photoshop_(medium)",
]

# From KohakuBlueleaf/HakuBooru
fav_count_percentile_full = {
    "g": {
        5: 1,
        10: 1,
        15: 2,
        20: 3,
        25: 3,
        30: 4,
        35: 5,
        40: 6,
        45: 7,
        50: 8,
        55: 9,
        60: 10,
        65: 12,
        70: 14,
        75: 16,
        80: 18,
        85: 22,
        90: 27,
        95: 37,
    },
    "s": {
        5: 1,
        10: 2,
        15: 4,
        20: 5,
        25: 6,
        30: 8,
        35: 9,
        40: 11,
        45: 13,
        50: 15,
        55: 17,
        60: 19,
        65: 22,
        70: 26,
        75: 30,
        80: 36,
        85: 44,
        90: 56,
        95: 81,
    },
    "q": {
        5: 4,
        10: 8,
        15: 11,
        20: 14,
        25: 18,
        30: 21,
        35: 25,
        40: 29,
        45: 33,
        50: 38,
        55: 43,
        60: 49,
        65: 56,
        70: 65,
        75: 75,
        80: 88,
        85: 105,
        90: 132,
        95: 182,
    },
    "e": {
        5: 4,
        10: 9,
        15: 13,
        20: 18,
        25: 22,
        30: 27,
        35: 33,
        40: 39,
        45: 45,
        50: 52,
        55: 60,
        60: 69,
        65: 79,
        70: 92,
        75: 106,
        80: 125,
        85: 151,
        90: 190,
        95: 262,
    },
}


def get_quality_tag(score, rating):
    percentile = fav_count_percentile_full[rating]
    if score > percentile[95]:
        quality_tag = "best quality"
    elif score > percentile[90]:
        quality_tag = "high quality"
    elif score > percentile[75]:
        quality_tag = "great quality"
    elif score > percentile[50]:
        quality_tag = "medium quality"
    elif score > percentile[25]:
        quality_tag = "normal quality"
    elif score > percentile[10]:
        quality_tag = "bad quality"
    elif score > percentile[5]:
        quality_tag = "low quality"
    else:
        quality_tag = "worst quality"
    return quality_tag


def get_aesthetic_tag(score):
    if score > 1.50: # out of the scale
        return "out of the scale aesthetic"
    if score > 1.10: # out of the scale
        return "masterpiece"
    elif score > 0.92:
        return "extremely aesthetic"
    elif score > 0.85:
        return "very aesthetic"
    elif score > 0.75:
        return "aesthetic"
    elif score > 0.50:
        return "slightly aesthetic"
    elif score > 0.40:
        return "not aesthetic"
    elif score > 0.30:
        return "not displeasing"
    elif score > 0.20:
        return "slightly displeasing"
    elif score > 0.10:
        return "displeasing"
    else:
        return "very displeasing"


def get_tags_from_json(json_path):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
    line = f"year {json_data['created_at'][:4]}"
    for tag in json_data["tag_string_general"].split(" "):
        if tag:
            line += f", {tag.replace('_', ' ') if len(tag) > 3 else tag}"
    for meta_tag in json_data["tag_string_meta"].split(" "):
        if meta_tag and not any([bool(meta_tag_blacklist in meta_tag) for meta_tag_blacklist in meta_blacklist]):
            line += f", {meta_tag.replace('_', ' ')}"
    if json_data["rating"] == "g":
        line += ", sfw"
    elif json_data["rating"] == "s":
        line += ", suggestive"
    elif json_data["rating"] == "q":
        line += ", nsfw"
    elif json_data["rating"] == "e":
        line += ", explicit nsfw"
    for char in json_data["tag_string_character"].split(" "):
        if char:
            line += f", character {char.replace('_', ' ')}"
    for cpr in json_data["tag_string_copyright"].split(" "):
        if cpr:
            line += f", from {cpr.replace('_', ' ')}"
    for artist in json_data["tag_string_artist"].split(" "):
        if artist:
            line += f", art by {artist.replace('_', ' ')}"
    line += ", " + get_quality_tag(json_data.get("fav_count", json_data["score"]), json_data["rating"])
    if json_data.get("aesthetic-shadow-v2", None) is not None:
        line += ", " + get_aesthetic_tag(json_data["aesthetic-shadow-v2"])
    return line


class ImageBackend():
    def __init__(self, batches, processor, load_queue_lenght=32, max_load_workers=4):
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
                time.sleep(0.25)
            elif not self.batches.empty():
                batches = self.batches.get()
                images = []
                image_paths = []
                prompts = []
                for batch in batches:
                    image, image_path, prompt = self.load_from_file(batch)
                    images.append(image)
                    image_paths.append(image_path)
                    prompts.append(prompt)
                inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding="longest", max_length=769, truncation=True) # 769 = 577 image + 192 text. max_length > 769 is too slow.
                attention_mask_image = torch.ones((inputs["attention_mask"].shape[0], 577), device=inputs["attention_mask"].device, dtype=inputs["attention_mask"].dtype)
                inputs["attention_mask"] = torch.cat([attention_mask_image, inputs["attention_mask"]], dim=1) # add atten mask for the image
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype, memory_format=torch.channels_last)
                self.load_queue.put([inputs, image_paths])
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")
        return


    def load_from_file(self, image_path):
        base_prompt = "Describe this anime image. Describe the overall art style and the quality of this anime image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well. Pretend that anime and cartoon is the reality. If you are not sure about something, then don't mention it."
        json_path = os.path.splitext(image_path)[0]+".json"
        if os.path.exists(json_path):
            prompt = f"{base_prompt} These are the tags for the anime image, you can use them for guidence: {get_tags_from_json(json_path)}"
        else:
            prompt = base_prompt
        image = Image.open(image_path).convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        return [image, image_path, prompt]


class SaveCaptionBackend():
    def __init__(self, processor, max_save_workers=2):
        self.processor = processor
        self.keep_saving = True
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)


    def save(self, generated_ids, image_paths):
        self.save_queue.put([generated_ids, image_paths])


    def save_thread_func(self):
        while self.keep_saving:
            if not self.save_queue.empty():
                generated_ids, image_paths = self.save_queue.get()
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                for i in range(len(image_paths)):
                    self.save_to_file(generated_text[i], os.path.splitext(image_paths[i])[0]+".txt")
            else:
                time.sleep(0.1)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        caption_file = open(path, "w")
        caption_file.write(data)
        caption_file.close()


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2" if use_flash_atten else None,
    ).to(device, dtype=dtype, memory_format=torch.channels_last).eval()
    model.requires_grad_(False)
    model.vision_tower.to(memory_format=torch.channels_last).eval()
    model.vision_tower.requires_grad_(False)
    model.language_model.to(memory_format=torch.channels_last).eval()
    model.language_model.requires_grad_(False)

    if "xpu" in device:
        model = ipex.llm.optimize(model, device=device, dtype=dtype, inplace=True)
    else:
        #torch.cuda.tunable.enable(val=True)
        model = torch.compile(model, mode="max-autotune", backend="inductor")


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
    image_backend = ImageBackend(batches, processor)
    save_backend = SaveCaptionBackend(processor)

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
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    pixel_values=inputs["pixel_values"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=3
                )
                save_backend.save(generated_ids, image_paths)
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

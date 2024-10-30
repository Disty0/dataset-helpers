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
from queue import Queue
from transformers import AutoModelForCausalLM, AutoProcessor
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

batch_size = 8
image_ext = ".jxl"
model_id = "MiaoshouAI/Florence-2-base-PromptGen-v1.5"
revision = "c06a5f02cc6071a5d65ee5d294cf3732d3097540"
device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
dtype = torch.float16 if "cuda" in device else torch.bfloat16 if "xpu" in device else torch.float32
use_flash_atten = "cuda" in device
steps_after_gc = -1

if image_ext == ".jxl":
    import pillow_jxl # noqa: F401
from PIL import Image # noqa: E402
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


if not use_flash_atten:
    try:
        import transformers
        from transformers.dynamic_module_utils import get_imports
        def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
            if not str(filename).endswith("modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            try:
                imports.remove("flash_attn")
            except Exception:
                pass
            return imports
        transformers.dynamic_module_utils.get_imports = fixed_get_imports
    except Exception:
        pass


meta_blacklist = [
    "highres",
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


danbooru_quality_scores = {
    "g": {
        1: 1,
        2: 3,
        3: 7,
        4: 15,
        5: 23,
        6: 35,
    },
    "s": {
        1: 4,
        2: 8,
        3: 16,
        4: 32,
        5: 48,
        6: 80,
    },
    "q": {
        1: 8,
        2: 12,
        3: 24,
        4: 48,
        5: 80,
        6: 136,
    },
    "e": {
        1: 12,
        2: 24,
        3: 50,
        4: 100,
        5: 160,
        6: 240,
    },
}


quality_score_to_tag = {
    6: "best quality",
    5: "high quality",
    4: "great quality",
    3: "normal quality",
    2: "bad quality",
    1: "low quality",
    0: "worst quality",
}


def get_quality_score_from_rating(score, rating):
    if score > danbooru_quality_scores[rating][6]:
        return 6
    elif score > danbooru_quality_scores[rating][5]:
        return 5
    elif score > danbooru_quality_scores[rating][4]:
        return 4
    elif score > danbooru_quality_scores[rating][3]:
        return 3
    elif score > danbooru_quality_scores[rating][2]:
        return 2
    elif score > danbooru_quality_scores[rating][1]:
        return 1
    else:
        return 0


def get_quality_tag_from_wd(score):
    if score > 0.97:
        return 6
    elif score > 0.92:
        return 5
    elif score > 0.75:
        return 4
    elif score > 0.30:
        return 3
    elif score > 0.08:
        return 2
    elif score > 0.01:
        return 1
    else:
        return 0


def get_quality_tag(json_data):
    quality_score = get_quality_score_from_rating(json_data.get("fav_count", json_data["score"]), json_data["rating"])
    if int(json_data["id"]) > 7000000:
        wd_quality_score = get_quality_tag_from_wd(json_data.get("wd-aes-b32-v0", 0))
        quality_score = max(quality_score, wd_quality_score)
    return quality_score_to_tag[quality_score]


def get_aesthetic_tag(score):
    if score > 0.925:
        return "very aesthetic"
    elif score > 0.90:
        return "highly aesthetic"
    elif score > 0.875:
        return "slightly aesthetic"
    elif score > 0.825:
        return "moderate aesthetic"
    elif score > 0.725:
        return "not aesthetic"
    else:
        return "bad aesthetic"


def get_tags_from_json(json_path):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
    if not json_data.get("florence-2-base-promptgen-v1-5", ""):
        return ""
    line = f"year {json_data['created_at'][:4]}"
    if json_data.get("special_tags", ""):
        for special_tag in json_data["special_tags"].split(" "):
            if special_tag:
                line += f", {special_tag.replace('_', ' ')}"
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
    if json_data.get('aesthetic-shadow-v2', None) is not None:
        line += f", {get_aesthetic_tag(json_data['aesthetic-shadow-v2'])}"
    line += f", {get_quality_tag(json_data)}"
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
        #base_prompt = "Describe this image."
        #base_prompt = "Describe this image in detail."
        #base_prompt = "Describe this image. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well."
        #base_prompt = "Describe this image. Describe the overall art style and the quality of this image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well."
        #base_prompt = "Describe this image. Describe the overall art style and the quality of this image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well. Pretend that anime and cartoon is the reality."
        #base_prompt = "Describe this image. Describe the overall art style and the quality of this anime image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well. Pretend that anime and cartoon is the reality."
        prompt = "Describe this anime image. Describe the overall art style and the quality of this anime image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well. Pretend that anime and cartoon is the reality. If you are not sure about something, then don't mention it."
        json_path = os.path.splitext(image_path)[0]+".json"
        if os.path.exists(json_path):
            booru_tags = get_tags_from_json(json_path)
            if booru_tags:
                prompt += " These are the tags for the anime image, you can use them for guidence: " + booru_tags
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
                    self.save_to_file(generated_text[i], os.path.splitext(image_paths[i])[0]+".json")
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        with open(path, "r") as f:
            json_data = json.load(f)
        json_data["florence-2-base-promptgen-v1-5"] = data.split("\n", maxsplit=1)[0].replace("\r", "")
        with open(path, "w") as f:
            json.dump(json_data, f)


if __name__ == '__main__':
    try:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
    except Exception:
        pass
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2" if use_flash_atten else None,
    ).to(device, dtype=dtype, memory_format=torch.channels_last).eval()
    model.requires_grad_(False)
    model.vision_tower.to(memory_format=torch.channels_last).eval()
    model.vision_tower.requires_grad_(False)
    model.language_model.eval()
    model.language_model.requires_grad_(False)

    if "xpu" in device:
        #model.vision_tower = ipex.llm.optimize(model.vision_tower, device=device, dtype=dtype, inplace=True)
        #model.language_model = ipex.llm.optimize(model.language_model, device=device, dtype=dtype, inplace=True)
        pass
    else:
        #torch.cuda.tunable.enable(val=True)
        model.vision_tower = torch.compile(model.vision_tower, mode="max-autotune", backend="inductor")
        model.language_model = torch.compile(model.language_model, mode="max-autotune", backend="inductor")


    print(f"Searching for {image_ext} files...")
    file_list = glob.glob(f'./**/*{image_ext}')
    image_paths = []

    for image_path in tqdm(file_list):
        try:
            json_path = os.path.splitext(image_path)[0]+".json"
            with open(json_path, "r") as f:
                json_data = json.load(f)
            if (not json_data.get("florence-2-base-promptgen-v1-5", "")
                or "1girl" in json_data["florence-2-base-promptgen-v1-5"]
                or "2girl" in json_data["florence-2-base-promptgen-v1-5"]
                or "3girl" in json_data["florence-2-base-promptgen-v1-5"]
                or "4girl" in json_data["florence-2-base-promptgen-v1-5"]
                or "1boy" in json_data["florence-2-base-promptgen-v1-5"]
                or "2boy" in json_data["florence-2-base-promptgen-v1-5"]
                or "3boy" in json_data["florence-2-base-promptgen-v1-5"]
                or "4boy" in json_data["florence-2-base-promptgen-v1-5"]
                or ", multiple girls," in json_data["florence-2-base-promptgen-v1-5"]
                or ", multiple boys," in json_data["florence-2-base-promptgen-v1-5"]
                or "\\)" in json_data["florence-2-base-promptgen-v1-5"]
                or "\\(" in json_data["florence-2-base-promptgen-v1-5"]
            ):
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

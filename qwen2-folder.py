#!/usr/bin/env python

import os
import gc
import math
import glob
import json
import time
import atexit
import torch
try:
    #import intel_extension_for_pytorch as ipex
    import ipex_llm
except Exception:
    pass
from PIL import Image
from queue import Queue
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

batch_size = 1
image_ext = ".webp"
max_image_size = 1048576 # 1024x1024
model_id = "Ertugrul/Qwen2-VL-7B-Captioner-Relaxed"
device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
dtype = torch.bfloat16
use_flash_atten = "cuda" in device and torch.version.cuda
steps_after_gc = -1


if not use_flash_atten:
    try:
        import transformers
        from transformers.dynamic_module_utils import get_imports
        def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
            if not str(filename).endswith("modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            imports.remove("flash_attn")
            return imports
        transformers.dynamic_module_utils.get_imports = fixed_get_imports
    except Exception:
        pass
if torch.version.hip:
    try:
        # don't use this for training models, only for inference with latent encoder and embed encoder
        # https://github.com/huggingface/diffusers/discussions/7172
        from functools import wraps
        from flash_attn import flash_attn_func
        backup_sdpa = torch.nn.functional.scaled_dot_product_attention
        @wraps(torch.nn.functional.scaled_dot_product_attention)
        def sdpa_hijack(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if query.shape[-1] <= 128 and attn_mask is None and query.dtype != torch.float32:
                return flash_attn_func(q=query.transpose(1, 2), k=key.transpose(1, 2), v=value.transpose(1, 2), dropout_p=dropout_p, causal=is_causal, softmax_scale=scale).transpose(1, 2)
            else:
                return backup_sdpa(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        torch.nn.functional.scaled_dot_product_attention = sdpa_hijack
    except Exception as e:
        print(f"Failed to enable Flash Atten for ROCm: {e}")



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


quality_score_to_tag = {
    7: "best quality",
    6: "high quality",
    5: "great quality",
    4: "medium quality",
    3: "normal quality",
    2: "bad quality",
    1: "low quality",
    0: "worst quality",
}


def get_quality_score_from_rating(score, rating):
    percentile = fav_count_percentile_full[rating]
    if score > percentile[95]:
        return 7
    elif score > percentile[90]:
        return 6
    elif score > percentile[75]:
        return 5
    elif score > percentile[50]:
        return 4
    elif score > percentile[25]:
        return 3
    elif score > percentile[10]:
        return 2
    elif score > percentile[5]:
        return 1
    else:
        return 0

def get_quality_tag_from_wd(score):
    if score > 0.98:
        return 7
    elif score > 0.90:
        return 6
    elif score > 0.75:
        return 5
    elif score > 0.50:
        return 4
    elif score > 0.25:
        return 3
    elif score > 0.125:
        return 2
    elif score > 0.025:
        return 1
    else:
        return 0

def get_quality_tag(json_data):
    quality_score = get_quality_score_from_rating(json_data.get("fav_count", json_data["score"]), json_data["rating"])
    if json_data["id"] > 7000000:
        wd_quality_score = get_quality_tag_from_wd(json_data.get("wd-aes-b32-v0", 0))
        quality_score = max(quality_score, wd_quality_score)
    return quality_score_to_tag[quality_score]


def get_aesthetic_tag(score):
    if score > 0.92:
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
    copyright_tags = ""
    for char in json_data["tag_string_character"].split(" "):
        if char:
            copyright_tags += f", character {char.replace('_', ' ')}"
    for cpr in json_data["tag_string_copyright"].split(" "):
        if cpr:
            copyright_tags += f", from {cpr.replace('_', ' ')}"
    for artist in json_data["tag_string_artist"].split(" "):
        if artist:
            copyright_tags += f", art by {artist.replace('_', ' ')}"
    if copyright_tags:
        copyright_tags = copyright_tags[2:]
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
    if json_data.get('aesthetic-shadow-v2', None) is not None:
        line += f", {get_aesthetic_tag(json_data['aesthetic-shadow-v2'])}"
    line += f", {get_quality_tag(json_data)}"
    return line, copyright_tags


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
                copyright_tags = ""
                for batch in batches:
                    image, image_path, prompt, char_tag = self.load_from_file(batch)
                    images.append(image)
                    image_paths.append(image_path)
                    prompts.append(prompt)
                    if char_tag:
                        copyright_tags += " " + char_tag
                if copyright_tags:
                    copyright_tags = copyright_tags[1:].replace(" character ", " ").replace(" from ", " ").replace(" art by ", " ").replace(", ", " ")
                inputs = self.processor(text=prompts, images=images, padding="longest", return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
                self.load_queue.put([inputs, image_paths, copyright_tags])
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")
        return


    def load_from_file(self, image_path):
        copyright_tags = ""
        prompt = "Describe this anime image in detail. Describe the art style, anime style and the quality of this anime image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions as well. Pretend that anime and cartoon is the reality. Don't overdetail the caption."
        json_path = os.path.splitext(image_path)[0]+".json"
        if os.path.exists(json_path):
            booru_tags, copyright_tags = get_tags_from_json(json_path)
            if copyright_tags:
                prompt += " These are the character, series and artist names for this anime image, use them: " + copyright_tags + "."
            prompt += " These are the tags for this anime image, you can use them for guidence: " + booru_tags
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        image = Image.open(image_path)
        width, height = image.size
        image_size = width * height
        if image_size > max_image_size:
            scale = math.sqrt(image_size / max_image_size)
            new_width = int(width/scale)
            new_height = int(height/scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
        return [image, image_path, text_prompt, copyright_tags]


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
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for i in range(len(image_paths)):
                    generated_text[i] = generated_text[i].replace("Describe this anime image in detail.", "").replace("Describe the anime style, anime style and the quality of this anime image as well.", "").replace("Describe nudity, sex, sexual intercourse, sex", "")
                    self.save_to_file(generated_text[i], os.path.splitext(image_paths[i])[0]+".txt")
            else:
                time.sleep(0.1)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        caption_file = open(path, "w")
        caption_file.write(data)
        caption_file.close()


# qwen2 stops generating when it writes a coptyright name
# blackist the end and the image pad tokens when the generation lenght is too short or the last word is a copyright
class UncensorQwen2(LogitsProcessor):
    def __call__(self, input_ids, scores):
        global input_ids_len, copyright_tags, processor
        generation_lenght = input_ids.shape[-1] - input_ids_len
        if generation_lenght < 64:
            for i in range(scores.shape[0]):
                scores[i][151645] = 0
                scores[i][151655] = 0
        else:
            for i in range(scores.shape[0]):
                max_id = torch.argmax(scores)
                if (copyright_tags and (max_id == 151645 or max_id == 151655) and generation_lenght < 384
                and processor.decode(input_ids[i][-25:]).lower().split("\n")[-1].split(" ")[-1] in copyright_tags):
                        scores[i][151645] = 0
                        scores[i][151655] = 0
                elif max_id == 151655:
                    # stop if the next token is the image pad
                    # qwen2 spams this token until it hits the max token limit
                    scores[i][151645] = scores[0][151655]
                    scores[i][151655] = 0
        return scores


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained(model_id)
    logits_processor = UncensorQwen2()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device if "xpu" not in device else "cpu",
        attn_implementation="flash_attention_2" if use_flash_atten else None,
    ).to(dtype=dtype).eval()
    model.requires_grad_(False)
    model.visual.eval()
    model.visual.requires_grad_(False)
    model.model.eval()
    model.model.requires_grad_(False)

    if "xpu" in device:
        model = ipex_llm.optimize_model(model, low_bit="sym_int8")
        model = model.to(device)


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
                inputs, image_paths, copyright_tags = image_backend.get_images()
                inputs = inputs.to(device)
                input_ids_len = inputs["input_ids"].shape[-1]
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    use_cache=True,
                    logits_processor=[logits_processor],
                )
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                save_backend.save(generated_ids, image_paths)
            except Exception as e:
                os.makedirs("errors", exist_ok=True)
                error_file = open("errors/errors.txt", 'a')
                error_file.write(f"ERROR: {image_paths} MESSAGE: {e} \n")
                error_file.close()
            steps_after_gc = steps_after_gc + 1
            if steps_after_gc == 0 or steps_after_gc >= 256 if "xpu" not in device else 1:
                if "cpu" not in device:
                    getattr(torch, torch.device(device).type).synchronize()
                    getattr(torch, torch.device(device).type).empty_cache()
                gc.collect()
                steps_after_gc = 1 if steps_after_gc == 0 else 0

    atexit.unregister(exit_handler)
    exit_handler(image_backend, save_backend)
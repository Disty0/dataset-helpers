#!/usr/bin/env python

import os
import gc
import json
import time
import atexit
import random

import torch
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

try:
    import transformers # ipex hijacks transformers and makes it unable to load a model
    backup_get_class_from_dynamic_module = transformers.dynamic_module_utils.get_class_from_dynamic_module
    import intel_extension_for_pytorch as ipex
    ipex_available = True
    ipex.llm.utils._get_class_from_dynamic_module = backup_get_class_from_dynamic_module
    transformers.dynamic_module_utils.get_class_from_dynamic_module = backup_get_class_from_dynamic_module
except Exception:
    ipex_available = False


from queue import Queue
from transformers import AutoModelForCausalLM, AutoProcessor
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tqdm import tqdm

try:
    import pillow_jxl # noqa: F401
except Exception:
    pass
from PIL import Image # noqa: E402

from typing import Dict, List, Tuple, Union

batch_size = 8
use_tunable_ops = False
use_torch_compile = False
caption_key = "florence-2-base-promptgen-v1-5"
model_id = "MiaoshouAI/Florence-2-base-PromptGen-v1.5"
revision = "c06a5f02cc6071a5d65ee5d294cf3732d3097540"
device = torch.device("cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.bfloat16 if device.type == "xpu" else torch.float32
use_flash_atten = device.type == "cuda"
img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
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


meta_blacklist = (
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
    "_file",
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
    "spoilers",
    "commission",
)


style_age_tags = (
    "1920s_(style)",
    "1930s_(style)",
    "1950s_(style)",
    "1960s_(style)",
    "1970s_(style)",
    "1980s_(style)",
    "1990s_(style)",
    "2000s_(style)",
    "2010s_(style)",
    "2015s_(style)",
    "2020s_(style)",
)


no_shuffle_tags = (
    "1girl",
    "2girls",
    "3girls",
    "4girls",
    "5girls",
    "6+girls",
    "multiple_girls",
    "1boy",
    "2boys",
    "3boys",
    "4boys",
    "5boys",
    "6+boys",
    "multiple_boys",
    "male_focus",
    "1other",
    "2others",
    "3others",
    "4others",
    "5others",
    "6+others",
    "multiple_others",
)


danbooru_quality_scores = {
    "g": {6: 50, 5: 30, 4: 20, 3: 10, 2: 5, 1: 1},
    "s": {6: 150, 5: 80, 4: 50, 3: 20, 2: 10, 1: 5},
    "q": {6: 300, 5: 200, 4: 100, 3: 50, 2: 25, 1: 10},
    "e": {6: 420, 5: 280, 4: 180, 3: 100, 2: 50, 1: 25}
}


aes_wd14_scores = {6: 0.999666, 5: 0.9983, 4: 0.992, 3: 0.50, 2: 0.016, 1: 0.0002}
aes_shadow_scores = {6: 0.938, 5: 0.925, 4: 0.911, 3: 0.875, 2: 0.825, 1: 0.750}
aes_deepghs_scores = {6: 0.962, 5: 0.890, 4: 0.786, 3: 0.585, 2: 0.388, 1: 0.192}
aes_euge_scores = {6: 0.8396, 5: 0.7405, 4: 0.6942, 3: 0.3698, 2: 0.2940, 1: 0.1569}


quality_score_to_tag = {
    6: "best quality",
    5: "high quality",
    4: "great quality",
    3: "normal quality",
    2: "low quality",
    1: "bad quality",
    0: "worst quality",
}


aes_score_to_tag = {
    6: "very aesthetic", # less than 1000 images are able to get this score when using multiple aes models
    5: "very aesthetic",
    4: "highly aesthetic",
    3: "moderate aesthetic",
    2: "low aesthetic",
    1: "bad aesthetic",
    0: "worst aesthetic",
}


def get_aes_score(score: int, score_dict: Dict[int, int]) -> int:
    for i in reversed(range(6)):
        if score > score_dict[i+1]:
            return i+1
    return 0


def get_combined_aes_score(scores: List[int], score_dicts: List[Dict[int, int]]) -> int:
    combined_score = 0
    for score in scores:
        combined_score += score
    combined_score_dict = {6:0, 5:0, 4:0, 3:0, 2:0, 1:0}
    for score_dict in score_dicts:
        for key, score in score_dict.items():
            combined_score_dict[key] += score
    return get_aes_score(combined_score, combined_score_dict)


def get_aesthetic_tag(json_data: Dict[str, int]) -> str:
    scores = []
    score_dicts = []
    if json_data.get("wd-aes-b32-v0", None) is not None:
        scores.append(json_data["wd-aes-b32-v0"])
        score_dicts.append(aes_wd14_scores)
    if json_data.get("aesthetic-shadow-v2", None) is not None:
        scores.append(json_data["aesthetic-shadow-v2"])
        score_dicts.append(aes_shadow_scores)
    if json_data.get("swinv2pv3_v0_448_ls0.2_x_percentile", None) is not None:
        scores.append(json_data["swinv2pv3_v0_448_ls0.2_x_percentile"])
        score_dicts.append(aes_deepghs_scores)
    if json_data.get("waifu-scorer-v3", None) is not None:
        scores.append(json_data["waifu-scorer-v3"])
        score_dicts.append(aes_euge_scores)
    if len(scores) == 1:
        print(f"Using only 1 AES score! ID: {json_data.get('id', 'none')}")
        aes_score = get_aes_score(scores[0], score_dicts[0])
    else:
        aes_score = get_combined_aes_score(scores, score_dicts)
    return aes_score_to_tag[aes_score]


def get_quality_tag(json_data: Dict[str, int]) -> str:
    if json_data.get("fav_count", None) is not None or json_data.get("score", None) is not None:
        quality_score = get_aes_score(
            json_data.get("fav_count", json_data["score"]),
            danbooru_quality_scores[json_data.get("wd_rating", json_data["rating"])]
        )
        if int(json_data["id"]) > 7000000:
            wd_quality_score = get_aes_score(json_data.get("swinv2pv3_v0_448_ls0.2_x_percentile", 0), aes_deepghs_scores)
            quality_score = max(quality_score, wd_quality_score)
    else:
        quality_score = get_aes_score(json_data["swinv2pv3_v0_448_ls0.2_x_percentile"], aes_deepghs_scores)
    return quality_score_to_tag[quality_score]


def dedupe_tags(split_tags: List[str]) -> List[str]:
    if len(split_tags) <= 1:
        return split_tags
    split_tags.sort(key=len, reverse=True)
    deduped_tags = []
    ordered_tag_string = ""
    for tag in split_tags:
        spaced_tag = "_" + tag + "_"
        if tag and spaced_tag not in ordered_tag_string and tag not in deduped_tags:
            ordered_tag_string += spaced_tag
            deduped_tags.append(tag)
    random.shuffle(deduped_tags)
    return deduped_tags


def dedupe_character_tags(split_tags: List[str]) -> List[str]:
    if len(split_tags) <= 1:
        return split_tags
    split_tags.sort(key=len, reverse=True)
    deduped_tags = []
    ordered_tag_string = ""
    for tag in split_tags:
        pruned_tag_end = ""
        pruned_tags = tag.rsplit("_(", maxsplit=1)
        if len(pruned_tags) > 1:
            pruned_tag, pruned_tag_end = pruned_tags
            pruned_tag_end = "_(" + pruned_tag_end
        else:
            pruned_tag = pruned_tags[0]
        spaced_tag = "_" + tag + "_"
        if tag and spaced_tag not in ordered_tag_string and tag not in deduped_tags and not (
        pruned_tag in ordered_tag_string and pruned_tag_end in ordered_tag_string):
            ordered_tag_string += spaced_tag
            deduped_tags.append(tag)
    random.shuffle(deduped_tags)
    return deduped_tags


def get_tags_from_json(json_path: str, image_path: str) -> str:
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    line = get_aesthetic_tag(json_data)
    line += f", {get_quality_tag(json_data)}"
    year_tag = str(json_data['created_at'][:4])
    line += f", year {year_tag}"

    style_age_tag_added = False
    split_general_tags = json_data["tag_string_general"].split(" ")
    for style_age_tag in style_age_tags:
        if style_age_tag in split_general_tags:
            split_general_tags.pop(split_general_tags.index(style_age_tag))
            if not style_age_tag_added and int(style_age_tag[:3]) < int(json_data['created_at'][:3]):
                line += f", {style_age_tag[:4]}s (style)"
                style_age_tag_added = True
    if (not style_age_tag_added and json_data.get("style_age", "")
        and (
            int(json_data['style_age'][:3]) < int(json_data['created_at'][:3])
            or ((2015 <= int(json_data['created_at'][:4]) < 2020) and int(json_data['style_age'][:4]) < 2015)
        )
    ):
        line += f", {json_data['style_age'][:4]}s (style)"

    if json_data.get("special_tags", ""):
        for special_tag in json_data["special_tags"].split(" "):
            if special_tag:
                line += f", {special_tag.replace('_', ' ')}"

    for artist in json_data["tag_string_artist"].split(" "):
        if artist:
            line += f", art by {artist.replace('_', ' ')}"

    split_meta_tags = json_data["tag_string_meta"].split(" ")
    random.shuffle(split_meta_tags)
    for medium_tag in json_data["tag_string_meta"].split(" "):
        if medium_tag.endswith("_(medium)") and medium_tag != "photoshop_(medium)":
            split_meta_tags.pop(split_meta_tags.index(medium_tag))
            line += f", {medium_tag.replace('_', ' ')}"

    rating = json_data.get("wd_rating", json_data["rating"])
    if rating == "g":
        line += ", sfw rating"
    elif rating == "s":
        line += ", sensitive rating"
    elif rating == "q":
        line += ", nsfw rating"
    elif rating == "e":
        line += ", explicit nsfw rating"

    for no_shuffle_tag in no_shuffle_tags:
        if no_shuffle_tag in split_general_tags:
            split_general_tags.pop(split_general_tags.index(no_shuffle_tag))
            line += f", {no_shuffle_tag.replace('_', ' ')}"

    for char in dedupe_character_tags(json_data["tag_string_character"].split(" ")):
        if char:
            line += f", character {char.replace('_', ' ')}"

    split_copyright_tags = json_data["tag_string_copyright"].split(" ")
    if "original" in split_copyright_tags:
        split_copyright_tags.pop(split_copyright_tags.index("original"))
    for cpr in dedupe_tags(split_copyright_tags):
        if cpr:
            line += f", from {cpr.replace('_', ' ')}"

    if json_data.get("wd_tag_string_general", ""):
        for wd_tag in json_data["wd_tag_string_general"].split(" "):
            if wd_tag and wd_tag not in no_shuffle_tags and wd_tag not in style_age_tags and wd_tag not in split_general_tags:
                split_general_tags.append(wd_tag)

    if json_data.get("file_ext", "jpg") not in {"png", "jxl"} and (json_data.get("file_size", float("inf")) < 307200 or os.path.getsize(image_path) < 307200):
        split_general_tags.append("compression_artifacts")

    for tag in dedupe_tags(split_general_tags):
        if tag:
            line += f", {tag.replace('_', ' ') if len(tag) > 3 else tag}"

    if split_meta_tags:
        for meta_tag in split_meta_tags:
            if meta_tag and not any([bool(meta_tag_blacklist in meta_tag) for meta_tag_blacklist in meta_blacklist]):
                line += f", {meta_tag.replace('_', ' ')}"

    return line


class ImageBackend():
    def __init__(self, batches: List[List[str]], processor: AutoProcessor, load_queue_lenght: int = 32, max_load_workers: int = 4):
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

    def get_images(self) -> Tuple[List[Dict[str, torch.Tensor]], List[str]]:
        result = self.load_queue.get()
        self.load_queue_lenght -= 1
        return result

    @torch.no_grad()
    def load_thread_func(self) -> None:
        while self.keep_loading:
            if self.load_queue_lenght >= self.max_load_queue_lenght:
                time.sleep(0.25)
            elif not self.batches.empty():
                batches = self.batches.get()
                images = []
                prompts = []
                for batch in batches:
                    image, prompt = self.load_from_file(batch)
                    images.append(image)
                    prompts.append(prompt)
                inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding="longest", max_length=769, truncation=True) # 769 = 577 image + 192 text. max_length > 769 is too slow.
                attention_mask_image = torch.ones((inputs["attention_mask"].shape[0], 577), device=inputs["attention_mask"].device, dtype=inputs["attention_mask"].dtype)
                inputs["attention_mask"] = torch.cat([attention_mask_image, inputs["attention_mask"]], dim=1) # add atten mask for the image
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
                self.load_queue.put((inputs, batches))
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")

    def load_from_file(self, image_path: str) -> Tuple[Image.Image, str]:
        prompt = "Describe this anime image. Describe the overall art style and the quality of this anime image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well. Pretend that anime and cartoon is the reality. If you are not sure about something, then don't mention it."
        #prompt = "Describe this image. Describe the overall art style and the quality of this anime image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well. Pretend that anime and cartoon is the reality."
        #prompt = "Describe this image. Describe the overall art style and the quality of this image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well. Pretend that anime and cartoon is the reality."
        #prompt = "Describe this image. Describe the overall art style and the quality of this image as well. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well."
        #prompt = "Describe this image. Describe nudity, sex, sexual intercourse, sex positions, bodily fluids, genitals, sexual acts, sexual interactions, intimacy, vulnerability and sensuality as well."
        #prompt = "Describe this image in detail."
        #prompt = "Describe this image."

        json_path = os.path.splitext(image_path)[0]+".json"
        if os.path.exists(json_path):
            booru_tags = get_tags_from_json(json_path, image_path)
            if booru_tags:
                prompt += " These are the tags for the anime image, you can use them for guidence: " + booru_tags
        image = Image.open(image_path).convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        return (image, prompt)


class SaveCaptionBackend():
    def __init__(self, processor: AutoProcessor, max_save_workers: int = 2):
        self.processor = processor
        self.keep_saving = True
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)

    def save(self, generated_ids: Union[List[List[int]], torch.Tensor], image_paths: List[str]) -> None:
        self.save_queue.put((generated_ids, image_paths))

    @torch.no_grad()
    def save_thread_func(self) -> None:
        while self.keep_saving:
            if not self.save_queue.empty():
                generated_ids, image_paths = self.save_queue.get()
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                for i in range(len(image_paths)):
                    self.save_to_file(generated_text[i], os.path.splitext(image_paths[i])[0]+".json")
            else:
                time.sleep(0.25)
        print("Stopping the save backend threads")

    def save_to_file(self, data: str, path: str) -> None:
        with open(path, "r") as f:
            json_data = json.load(f)
        json_data[caption_key] = data.split("\n", maxsplit=1)[0].replace("\r", "")
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

    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, torch_dtype=dtype,
        attn_implementation="flash_attention_2" if use_flash_atten else None,
    ).to(device, dtype=dtype).eval()
    model.requires_grad_(False)
    model.vision_tower.eval()
    model.vision_tower.requires_grad_(False)
    model.language_model.eval()
    model.language_model.requires_grad_(False)

    if ipex_available:
        model.vision_tower = ipex.llm.optimize(model.vision_tower, device=device, dtype=dtype, inplace=True)
        model.language_model = ipex.llm.optimize(model.language_model, device=device, dtype=dtype, inplace=True)
    elif use_torch_compile:
        model.vision_tower.forward_features_unpool = torch.compile(model.vision_tower.forward_features_unpool, backend="inductor")
        model.language_model.generate = torch.compile(model.language_model.generate, backend="inductor")


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
            if (not json_data.get(caption_key, "")
                or "1girl" in json_data[caption_key]
                or "2girl" in json_data[caption_key]
                or "3girl" in json_data[caption_key]
                or "4girl" in json_data[caption_key]
                or "1boy" in json_data[caption_key]
                or "2boy" in json_data[caption_key]
                or "3boy" in json_data[caption_key]
                or "4boy" in json_data[caption_key]
                or ", multiple girls," in json_data[caption_key]
                or ", multiple boys," in json_data[caption_key]
                or "\\)" in json_data[caption_key]
                or "\\(" in json_data[caption_key]
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

    for _ in tqdm(range(epoch_len)):
        try:
            torch.compiler.cudagraph_mark_step_begin()
            inputs, image_paths = image_backend.get_images()
            generated_ids = model.generate(
                input_ids=inputs["input_ids"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
                num_beams=3,
            )
            save_backend.save(generated_ids, image_paths)
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

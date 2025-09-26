#!/usr/bin/env python

import os
import gc
import time
import json
import atexit
import random
import argparse
from queue import Queue
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from typing import Dict, List


img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
tag_dict_path = os.path.join(os.path.dirname(__file__), "tag_dict.json")
tag_categories_path = os.path.join(os.path.dirname(__file__), "tag_categories.json")


if os.path.exists(tag_dict_path):
    with open(tag_dict_path, "r") as f:
        tag_dict = json.load(f)
else:
    tag_dict = None


if os.path.exists(tag_categories_path):
    with open(tag_categories_path, "r") as f:
        tag_categories = json.load(f)
else:
    tag_categories = None


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


pixiv_tag_blacklist = (
    "girl",
    "boy",
    "OC",
    "original",
    "original character",
    "illustration",
    "derivative work",
    "R-18",
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


def get_quality_tag(json_data: Dict[str, int], caption_key: str) -> str:
    if json_data.get("fav_count", None) is not None or json_data.get("score", None) is not None:
        quality_score = get_aes_score(
            json_data.get("fav_count", json_data["score"]),
            danbooru_quality_scores[json_data.get(f"{caption_key}_rating", json_data["rating"])]
        )
        if int(json_data["id"]) > 7000000:
            wd_quality_score = get_aes_score(json_data.get("swinv2pv3_v0_448_ls0.2_x_percentile", 0), aes_deepghs_scores)
            quality_score = max(quality_score, wd_quality_score)
    else:
        quality_score = get_aes_score(json_data["swinv2pv3_v0_448_ls0.2_x_percentile"], aes_deepghs_scores)
    return quality_score_to_tag[quality_score]


def dedupe_tags(split_tags: List[str], no_shuffle: bool) -> List[str]:
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
    if no_shuffle:
        random.shuffle(deduped_tags)
    return deduped_tags


def dedupe_character_tags(split_tags: List[str], no_shuffle: bool) -> List[str]:
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
    if no_shuffle:
        random.shuffle(deduped_tags)
    return deduped_tags


def get_tags_from_json(json_path: str, image_path: str, caption_key: str, no_shuffle: bool, general_only: bool) -> str:
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    split_general_tags = json_data["tag_string_general"].split(" ")
    split_artist_tags = json_data["tag_string_artist"].split(" ")
    split_copyright_tags = json_data["tag_string_copyright"].split(" ")
    split_character_tags = json_data["tag_string_character"].split(" ")
    split_meta_tags = json_data["tag_string_meta"].split(" ")
    split_raw_meta_tags = json_data["tag_string_meta"].split(" ")

    pixiv_tags = json_data.get("pixiv_tags", [])
    if pixiv_tags:
        for raw_pixiv_tag in pixiv_tags:
            if raw_pixiv_tag and raw_pixiv_tag not in pixiv_tag_blacklist:
                if raw_pixiv_tag.lower().endswith("+_bookmarks"):
                    raw_pixiv_tag = raw_pixiv_tag.rsplit("_", maxsplit=2)[0]
                if tag_dict is not None:
                    pixiv_tag = tag_dict.get(raw_pixiv_tag, raw_pixiv_tag.replace(" ", "_").lower())
                else:
                    pixiv_tag = raw_pixiv_tag.replace(" ", "_").lower()
                if tag_categories is not None and pixiv_tag.isascii():
                    pixiv_tag_category = tag_categories.get(pixiv_tag, 0)
                    if pixiv_tag_category == 0 and pixiv_tag not in split_general_tags:
                        split_general_tags.append(pixiv_tag)
                    elif pixiv_tag_category == 3 and pixiv_tag not in split_copyright_tags:
                        split_copyright_tags.append(pixiv_tag)
                    elif pixiv_tag_category == 4 and pixiv_tag not in split_character_tags:
                        split_character_tags.append(pixiv_tag)
                    elif pixiv_tag_category == 5 and pixiv_tag not in split_meta_tags:
                        split_meta_tags.append(pixiv_tag)
                        split_raw_meta_tags.append(pixiv_tag)
                elif pixiv_tag.isascii():
                    split_general_tags.append(pixiv_tag)

    if general_only:
        line = ""
    else:
        line = get_aesthetic_tag(json_data)
        line += f", {get_quality_tag(json_data, caption_key)}"
        year_tag = str(json_data['created_at'][:4])
        line += f", year {year_tag}"

    style_age_tag_added = False
    for style_age_tag in style_age_tags:
        if style_age_tag in split_general_tags:
            split_general_tags.pop(split_general_tags.index(style_age_tag))
            if not general_only:
                if not style_age_tag_added and int(style_age_tag[:3]) < int(json_data['created_at'][:3]):
                    line += f", {style_age_tag[:4]}s (style)"
                    style_age_tag_added = True
    if not general_only and (
        not style_age_tag_added and json_data.get("style_age", "") and (
            int(json_data['style_age'][:3]) < int(json_data['created_at'][:3])
            or ((2015 <= int(json_data['created_at'][:4]) < 2020) and int(json_data['style_age'][:4]) < 2015)
        )
    ):
        line += f", {json_data['style_age'][:4]}s (style)"

    if json_data.get("special_tags", ""):
        for special_tag in json_data["special_tags"].split(" "):
            if special_tag:
                line += f", {special_tag.replace('_', ' ')}"

    if not general_only:
        for artist in split_artist_tags:
            if artist:
                line += f", art by {artist.replace('_', ' ')}"

        if no_shuffle:
            random.shuffle(split_meta_tags)
        for medium_tag in split_raw_meta_tags:
            if medium_tag.endswith("_(medium)") and medium_tag != "photoshop_(medium)":
                split_meta_tags.pop(split_meta_tags.index(medium_tag))
                line += f", {medium_tag.replace('_', ' ')}"

        rating = json_data.get(f"{caption_key}_rating", json_data["rating"])
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

    for char in dedupe_character_tags(split_character_tags, no_shuffle):
        if char:
            line += f", character {char.replace('_', ' ')}"

    if "original" in split_copyright_tags:
        split_copyright_tags.pop(split_copyright_tags.index("original"))
    for cpr in dedupe_tags(split_copyright_tags, no_shuffle):
        if cpr:
            line += f", from {cpr.replace('_', ' ')}"

    if json_data.get(f"{caption_key}_tag_string_general", ""):
        for wd_tag in json_data[f"{caption_key}_tag_string_general"].split(" "):
            if wd_tag and wd_tag not in no_shuffle_tags and wd_tag not in style_age_tags and wd_tag not in split_general_tags:
                split_general_tags.append(wd_tag)

    if json_data.get("file_ext", "jpg") not in {"png", "jxl"} and (json_data.get("file_size", float("inf")) < 307200 or os.path.getsize(image_path) < 307200):
        split_general_tags.append("compression_artifacts")

    for tag in dedupe_tags(split_general_tags, no_shuffle):
        if tag:
            line += f", {tag.replace('_', ' ') if len(tag) > 3 else tag}"

    if not general_only and split_meta_tags:
        for meta_tag in split_meta_tags:
            if meta_tag and not any([bool(meta_tag_blacklist in meta_tag) for meta_tag_blacklist in meta_blacklist]):
                line += f", {meta_tag.replace('_', ' ')}"

    if general_only:
        line = line[2:]
    return line


class SaveTagBackend():
    def __init__(self, out_path: str, max_save_workers=8):
        self.out_path = out_path
        self.keep_saving = True
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)

    def save(self, data: str, path: str) -> None:
        self.save_queue.put((data,path))

    def save_thread_func(self) -> None:
        while self.keep_saving:
            if not self.save_queue.empty():
                data = self.save_queue.get()
                self.save_to_file(data[0], data[1])
            else:
                time.sleep(0.1)
        print("Stopping the save backend threads")

    def save_to_file(self, data: str, path: str) -> None:
        if self.out_path:
            os.makedirs(os.path.join(self.out_path, os.path.dirname(path)), exist_ok=True)
            caption_file = open(os.path.join(self.out_path, path), "w")
        else:
            caption_file = open(path, "w")
        caption_file.write(data)
        caption_file.close()


def main(out_path: str, caption_key: str, no_shuffle: bool, general_only: bool):
    steps_after_gc = 0
    print(f"Searching for {img_ext_list} files...")
    file_list = []
    for ext in img_ext_list:
        file_list.extend(glob(f"**/*.{ext}"))

    save_backend = SaveTagBackend(out_path, max_save_workers=4)

    def exit_handler(save_backend):
        while not save_backend.save_queue.empty():
            print(f"Waiting for the remaining writes: {save_backend.save_queue.qsize()}")
            time.sleep(1)
        save_backend.keep_saving = False
        save_backend.save_thread.shutdown(wait=True)
        del save_backend
    atexit.register(exit_handler, save_backend)

    for image_path in tqdm(file_list):
        json_path = os.path.splitext(image_path)[0]+".json"
        try:
            tags = get_tags_from_json(json_path, image_path, caption_key, no_shuffle, general_only)
            save_backend.save(tags, os.path.splitext(json_path)[0]+".txt")
        except Exception as e:
            os.makedirs("errors", exist_ok=True)
            error_file = open("errors/errors.txt", "a")
            error_file.write(f"ERROR: {json_path} MESSAGE: {e} \n")
            error_file.close()
        steps_after_gc = steps_after_gc + 1
        if steps_after_gc >= 100000:
            gc.collect()
            steps_after_gc = 0

    atexit.unregister(exit_handler)
    exit_handler(save_backend)

if __name__ == "__main__":
    #caption_keys: wd, pixiv
    parser = argparse.ArgumentParser(description="Create tags from json")
    parser.add_argument("--out_path", default="", type=str)
    parser.add_argument("--caption_key", default="wd", type=str)
    parser.add_argument("--no_shuffle", default=False, action="store_true")
    parser.add_argument("--general_only", default=False, action="store_true")
    args = parser.parse_args()
    main(args.out_path, args.caption_key, args.no_shuffle, args.general_only)

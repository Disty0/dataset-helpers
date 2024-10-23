#!/usr/bin/env python

import os
import gc
import glob
import time
import json
import atexit
from queue import Queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


image_ext = ".jxl"
out_path = ""
steps_after_gc = 0


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
    if json_data.get("score", None) is not None:
        quality_score = get_quality_score_from_rating(json_data.get("fav_count", json_data["score"]), json_data["rating"])
        if int(json_data["id"]) > 7000000:
            wd_quality_score = get_quality_tag_from_wd(json_data.get("wd-aes-b32-v0", 0))
            quality_score = max(quality_score, wd_quality_score)
    else:
        quality_score = get_quality_tag_from_wd(json_data.get("wd-aes-b32-v0", 0))
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
    line = get_aesthetic_tag(json_data['aesthetic-shadow-v2'])
    line += f", {get_quality_tag(json_data)}"
    line += f", year {json_data['created_at'][:4]}"
    if json_data.get("special_tags", ""):
        for special_tag in json_data["special_tags"].split(" "):
            if special_tag:
                line += f", {special_tag.replace('_', ' ')}"
    for artist in json_data["tag_string_artist"].split(" "):
        if artist:
            line += f", art by {artist.replace('_', ' ')}"
    for cpr in json_data["tag_string_copyright"].split(" "):
        if cpr:
            line += f", from {cpr.replace('_', ' ')}"
    for char in json_data["tag_string_character"].split(" "):
        if char:
            line += f", character {char.replace('_', ' ')}"
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
    return line


class SaveTagBackend():
    def __init__(self, max_save_workers=8):
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
                data = self.save_queue.get()
                self.save_to_file(data[0], data[1])
            else:
                time.sleep(0.1)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        if out_path:
            os.makedirs(os.path.join(out_path, os.path.dirname(path)), exist_ok=True)
            caption_file = open(os.path.join(out_path, path), "w")
        else:
            caption_file = open(path, "w")
        caption_file.write(data)
        caption_file.close()


if __name__ == '__main__':
    print(f"Searching for {image_ext} files...")
    file_list = glob.glob(f'**/*{image_ext}')

    save_backend = SaveTagBackend(max_save_workers=4)

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
            tags = get_tags_from_json(json_path)
            save_backend.save(tags, os.path.splitext(json_path)[0]+".txt")
        except Exception as e:
            os.makedirs("errors", exist_ok=True)
            error_file = open("errors/errors.txt", 'a')
            error_file.write(f"ERROR: {json_path} MESSAGE: {e} \n")
            error_file.close()
        steps_after_gc = steps_after_gc + 1
        if steps_after_gc >= 100000:
            gc.collect()
            steps_after_gc = 0

    atexit.unregister(exit_handler)
    exit_handler(save_backend)

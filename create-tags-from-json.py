#!/usr/bin/env python

import os
import gc
import glob
import time
import json
from queue import Queue
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

steps_after_gc = 0


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
    line = get_aesthetic_tag(json_data['aesthetic-shadow-v2'])
    line += f", {get_quality_tag(json_data)}"
    line += f", year {json_data['created_at'][:4]}"
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


if __name__ == '__main__':
    print("Searching for JSON files...")
    file_list = glob.glob('./**/*.json')

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
            caption_file = open(path, "w")
            caption_file.write(data)
            caption_file.close()


    save_backend = SaveTagBackend(max_save_workers=4)

    for json_path in tqdm(file_list):
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

    while not save_backend.save_queue.empty():
        print(f"Waiting for the remaining writes: {save_backend.save_queue.qsize()}")
        time.sleep(1)
    save_backend.keep_saving = False
    save_backend.save_thread.shutdown(wait=True)
    del save_backend

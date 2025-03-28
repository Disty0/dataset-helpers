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

from typing import Tuple

image_ext = ".jxl"
caption_key = "gemma-3-4b-it"
#caption_key = "qwen2.5-vl-7b-instruct"
#caption_key = "florence-2-base-promptgen-v1-5"
out_path = ""

is_gemma = "gemma" in caption_key

cleanup_start_list = (
    ["This is", ""],
    ["This ", ""],
    ["a highly detailed, high-quality anime image featuring ", ""],
    ["a highly detailed, high-quality ", "a "],
    ["a highly detailed, ", "a "],
    ["a highly detailed ", "a "],
    ["a high-quality, ", "a "],
    ["a high-quality ", "a "], # qwen2 slaps high-quality to everything even if it detects and captions quality issues correctly
    ["a and ", "a "],
    ["an anime image featuring ", ""],
    ["a anime image featuring ", ""],
    ["a anime", "an anime"],
    ["a aesthetic", "an aesthetic"],
    ["a explicit", "an explicit"],
)


cleanup_end_list = [
    "assistant",
    "the",
    "Describe",
    "the",
    "assistant",
    "image in",
    "image in detail",
    "bod",
    "genitals",
    "sexual",
    "sex positions",
    "positions",
    "sexual intercourse",
    "intercourse",
    "as well",
    "anime style and the quality of this image as well",
    "the",
]
cleanup_end_list.extend(cleanup_end_list)
cleanup_end_list.extend(cleanup_end_list)
cleanup_end_list = tuple(cleanup_end_list)

cleanup_caption_list = (
    [" Hatsune Maku, ", " Hatsune Miku, "],
    ["\"Vocaiol,\"", "\"Vocaloid,\""],
    ["Gensishn Impact", "Genshin Impact"],
    ["Gensishk Pnck", "Genshin Impact"],
    ["Gensish Pnol", "Genshin Impact"],
    ["Gensishenokai", "Genshin Impact"],
    ["Gensish Koyo", "Genshin Impact"],
    ["Gensish影响", "Genshin Impact"],
    ["Gensishnai Pnai", "Genshin Impact"],
    ["Azur_lane", "Azur Lane"],
    ["Kono Koushiki: Sekai ni Shukufuku o!", "Kono Subarashii Sekai ni Shukufuku wo!"],
    [" the the ", " the "],
    [" typical of high-quality ", " typical of "],
    [", based on the tags provided", ""],
    ["based on the tags provided", ""],
    [", based on the provided tags", ""],
    ["based on the provided tags", ""],
    [", as indicated by the tags", ""],
    ["as indicated by the tags", ""],
    ["Describe the", ""],
    ["Describe this", ""],
    ["Describe nudity", ""],
    ["Describe this anime", ""],
    ["Describe\nassistant", "\n"],
    ["\nassistant\n", "\n"],
    ["nsFW", "nsfw"],
    ['," ', '", '],
    ['." ', '". '],
    [",.", "."],
    [" ,", ","],
    [" .", "."],
    ["\n ", "\n"],
    [" \n", "\n"],
    # qwen2
    #####################################################
    # florence2
    [" vulnerability and vulnerability" , " vulnerability"],
    [" tranquility and tranquility" , " tranquility"],
    [" innocence and innocence" , " innocence"],
    [" introspection and introspection", " introspection"],
    [" emotional and emotional", "emotional"],
    [" isolation and isolation", " isolation"],
    [" readiness and readiness", " readiness"],
    [" slightly provocative and slightly provocative", " slightly provocative"],
    [" exaggerated features and exaggerated features", " exaggerated features"],
    [" slightly provocative and slightly provocative", " slightly provocative"],
    [" distractions or distractions", " distractions"],
    [" erect and slightly erect", " erect"],
    [" fantasy or fantasy", " fantasy"],
    [" enchantment and enchantment", " enchantment"],
    [" energetic and energetic", " energetic"],
    [" embarrassment and embarrassment", " embarrassment"],
    [" lively and lively", " lively"],
    [" allure and allure", " allure"],
    [" influence influence", " influence"],
)


def cleanup_word_repeats(caption: str) -> Tuple[str, str]:
    replace_words = None
    words = caption.split(" ")
    for i in range(len(words)):
        if words[-(2*i):-i] == words[-i:]:
            replace_words = words[-i:]
            break
    if replace_words is None:
        return caption, ""
    else:
        replace_string = " ".join(replace_words)
        if replace_string:
            while replace_string[0] == " ":
                replace_string = replace_string[1:]
            while replace_string[-1] == " ":
                replace_string = replace_string[:-1]
            while caption[-1] == " ":
                caption = caption[:-1]
            while caption.endswith(replace_string):
                caption = caption.removesuffix(replace_string)
                if caption:
                    while caption[-1] == " ":
                        caption = caption[:-1]
        return caption, replace_string


def cleanup_string_repeats(caption: str) -> Tuple[str, str]:
    replace_string = None
    for i in range(len(caption)):
        if caption[-(2*i):-i] == caption[-i:]:
            replace_string = caption[-i:]
            break
    if replace_string is None:
        return caption, ""
    else:
        if len(caption.rsplit(replace_string, maxsplit=3)) == 4:
            while caption.endswith(replace_string):
                caption = caption.removesuffix(replace_string)
            return caption, replace_string
        else:
            return caption, ""


def cleanup_word_repeats_recursive(caption: str) -> str:
    caption, replace_words = cleanup_word_repeats(caption)
    if replace_words:
        caption = cleanup_repeats_recursive(caption)
        caption = caption + " " + replace_words
        caption = cleanup_repeats_recursive(caption)
    return caption


def cleanup_repeats_recursive(caption: str) -> str:
    caption = cleanup_word_repeats_recursive(caption)
    caption, replace_string0 = cleanup_string_repeats(caption)
    if replace_string0:
        caption = caption + replace_string0
    caption = cleanup_word_repeats_recursive(caption)
    caption, replace_string1 = cleanup_string_repeats(caption)
    if replace_string1:
        caption = caption + replace_string1
        caption = cleanup_repeats_recursive(caption)
    return caption


def cleanup_whitespace(caption: str) -> str:
    while caption[0] == "\n":
        caption = caption[1:]
    while caption[0] == ",":
        caption = caption[1:]
    while caption[0] == ".":
        caption = caption[1:]
    while caption[0] == " ":
        caption = caption[1:]
    while caption[0] == "\n":
        caption = caption[1:]
    while caption[-1] == "\n":
        caption = caption[:-1]
    while caption[-1] == " ":
        caption = caption[:-1]
    while caption[-1] == ".":
        caption = caption[:-1]
    while caption[-1] == ",":
        caption = caption[:-1]
    while caption[-1] == "\n":
        caption = caption[:-1]
    return caption


def cleanup_caption(caption: str, json_data: dict = None) -> str:
    caption = cleanup_repeats_recursive(caption)
    caption = cleanup_whitespace(caption)
    for old_tag, new_tag in cleanup_caption_list:
        caption = caption.replace(old_tag, new_tag)
    caption = cleanup_whitespace(caption)
    for old_tag, new_tag in cleanup_start_list:
        if caption.startswith(old_tag):
            caption = new_tag + caption.removeprefix(old_tag)
    caption = cleanup_whitespace(caption)
    for old_tag in cleanup_end_list:
        while caption.endswith(old_tag):
            caption = caption.removesuffix(old_tag)
            caption = cleanup_whitespace(caption)
    if " tag" in caption.lower() and len(caption.split('", ')) > 5:
        caption = caption.replace('The image is tagged with ', '')
        caption = caption.replace('", and "', ', ')
        caption = caption.replace('" and "', ', ')
        caption = caption.replace('"', '')
        caption = caption.replace('\n ', '\n')
        caption = cleanup_whitespace(caption)
    if is_gemma:
        done_gemma_cleanup = False
        if caption.startswith("Here") or caption.startswith("Okay") or caption.startswith("here") or caption.startswith("okay"):
            split_caption = caption.split("\n", maxsplit=1)
            if len(split_caption) == 1:
                return ""
            else:
                caption = cleanup_whitespace(split_caption[-1])
            if caption.startswith("Here") or caption.startswith("Okay") or caption.startswith("here") or caption.startswith("okay"):
                split_caption = caption.split("\n", maxsplit=1)
                if len(split_caption) == 1:
                    return ""
                else:
                    caption = cleanup_whitespace(split_caption[-1])
            done_gemma_cleanup = True

        if (caption.startswith('"') and caption.endswith('"')) or (caption.startswith('“') and caption.endswith('”')):
            caption = caption[1:-1]
            done_gemma_cleanup = True

        split_caption = caption.rsplit("\n\n", maxsplit=2)
        if len(split_caption) > 2 and split_caption[-2] == "---":
            caption = cleanup_whitespace(split_caption[0])
            done_gemma_cleanup = True

        split_caption = caption.rsplit("\n", maxsplit=1)
        if split_caption[-1].startswith("I ") or split_caption[-1].startswith("I'"):
            caption = split_caption[0]
            done_gemma_cleanup = True
            split_caption = caption.rsplit("\n", maxsplit=1)
        elif split_caption[-1].startswith("Let ") or split_caption[-1].startswith("Let'") or split_caption[-1].startswith("Please let"):
            caption = split_caption[0]
            done_gemma_cleanup = True
            split_caption = caption.rsplit("\n", maxsplit=1)

        if split_caption[-1].endswith("?"):
            caption = split_caption[0]
            done_gemma_cleanup = True
        elif "disclaimer" in split_caption[-1].lower():
            caption = split_caption[0]
            done_gemma_cleanup = True
        elif "informational purposes" in split_caption[-1].lower():
            caption = split_caption[0]
            done_gemma_cleanup = True

        split_caption = caption.rsplit("\n**Tags:**\n", maxsplit=1)
        if len(split_caption) == 2:
            tags = split_caption[-1]
            tags = ("\n" + cleanup_whitespace(tags)).split("\n*   ")
            if len(tags) > 3:
                caption = split_caption[0] + "\n**Tags:**\n\n"
                for tag in tags:
                    if tag:
                        caption += tag.lower() + ", "
                caption = caption[:-2]
                done_gemma_cleanup = True

        if done_gemma_cleanup:
            caption = cleanup_whitespace(caption)
            caption = cleanup_repeats_recursive(caption)
            caption = cleanup_whitespace(caption)
    return caption


def get_captions_from_json(json_path: str) -> str:
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
    return cleanup_caption(json_data[caption_key], json_data=json_data)


class SaveTagBackend():
    def __init__(self, max_save_workers: int = 8):
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
        if data:
            if out_path:
                os.makedirs(os.path.join(out_path, os.path.dirname(path)), exist_ok=True)
                caption_file = open(os.path.join(out_path, path), "w")
            else:
                caption_file = open(path, "w")
            caption_file.write(data)
            caption_file.close()


def main():
    steps_after_gc = 0
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
            captions = get_captions_from_json(json_path)
            save_backend.save(captions, os.path.splitext(json_path)[0]+".txt")
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

if __name__ == '__main__':
    main()

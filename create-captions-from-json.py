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
caption_key = "qwen2-vl-7b-captioner-relaxed"
#caption_key = "florence-2-base-promptgen-v1-5"
out_path = ""
steps_after_gc = 0


cleanup_caption_list = [
    [" Hatsune Maku, ", " Hatsune Miku, "],
    ["\"Vocaiol,\"", "\"Vocaloid,\""],
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
]

def cleanup_whitespace(caption):
    while caption[0] == ",":
        caption = caption[1:]
    while caption[0] == ".":
        caption = caption[1:]
    while caption[0] == " ":
        caption = caption[1:]
    while caption[-1] == " ":
        caption = caption[:-1]
    while caption[-1] == ".":
        caption = caption[:-1]
    while caption[-1] == ",":
        caption = caption[:-1]
    return caption


def cleanup_caption(caption):
    caption = cleanup_whitespace(caption)
    for old_tag, new_tag in cleanup_caption_list:
        caption = caption.replace(old_tag, new_tag)
    caption = cleanup_whitespace(caption)
    return caption


def get_captions_from_json(json_path):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
    return cleanup_caption(json_data[caption_key])


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

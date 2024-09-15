#!/usr/bin/env python

import os
import gc
import glob
import time
import atexit
import torch
#try:
#    import intel_extension_for_pytorch as ipex # noqa: F401
#except Exception:
#    pass
from PIL import Image
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu" # else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available()
steps_after_gc = -1


def remove_old_tag(text):
    text = text.removeprefix("out of the scale aesthetic, ")
    text = text.removeprefix("masterpiece, ")
    text = text.removeprefix("extremely aesthetic, ")
    text = text.removeprefix("very aesthetic, ")
    text = text.removeprefix("asthetic, ")
    text = text.removeprefix("aesthetic, ")
    text = text.removeprefix("slightly asthetic, ")
    text = text.removeprefix("slightly aesthetic, ")
    text = text.removeprefix("not displeasing, ")
    text = text.removeprefix("not asthetic, ")
    text = text.removeprefix("not aesthetic, ")
    text = text.removeprefix("slightly displeasing, ")
    text = text.removeprefix("displeasing, ")
    text = text.removeprefix("very displeasing, ")
    return text



def get_aesthetic_tag(score):
    if score > 1.50: # out of the scale
        return "out of the scale aesthetic"
    if score > 1.10: # out of the scale
        return "masterpiece"
    elif score > 0.90:
        return "extremely aesthetic"
    elif score > 0.80:
        return "very aesthetic"
    elif score > 0.70:
        return "aesthetic"
    elif score > 0.50:
        return "slightly aesthetic"
    elif score > 0.40:
        return "not displeasing"
    elif score > 0.30:
        return "not aesthetic"
    elif score > 0.20:
        return "slightly displeasing"
    elif score > 0.10:
        return "displeasing"
    else:
        return "very displeasing"


class ImageBackend():
    def __init__(self, batches, load_queue_lenght=128, max_load_workers=4):
        self.load_queue_lenght = 0
        self.keep_loading = True
        self.batches = Queue()
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
                time.sleep(0.1)
            elif not self.batches.empty():
                batches = self.batches.get()
                current_batch = []
                for batch in batches:
                    current_batch.append(self.load_from_file(batch))
                self.load_queue.put(current_batch)
                self.load_queue_lenght += 1
            else:
                time.sleep(5)
        print("Stopping the image loader threads")
        return


    def load_from_file(self, image_path):
        image = Image.open(image_path).convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        return [image, image_path]


class SaveAestheticBackend():
    def __init__(self, max_save_workers=2):
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
        caption_file = open(path, "r")
        line = caption_file.readlines()[0]
        line = remove_old_tag(line)
        caption_file.close()
        caption_file = open(path, "w")
        caption_file.seek(0)
        caption_file.write(data + ", ")
        caption_file.write(line)
        caption_file.close()


if __name__ == '__main__':
    pipe = pipeline("image-classification", model="shadowlilac/aesthetic-shadow-v2", device=device)
    pipe.model.eval()
    pipe.model.requires_grad_(False)

    if "cpu" in device:
        os.environ.setdefault('PYTORCH_TRACING_MODE', 'TORCHFX')
        torch._dynamo.eval_frame.check_if_dynamo_supported = lambda: True
        import openvino.torch # noqa: F401
        pipe.model = torch.compile(pipe.model, backend='openvino', options = {"device" : "GPU", "model_caching": True, "cache_dir": os.path.join(os.getenv('HOME'), ".cache/openvino/model_cache")})
    #if "xpu" in device:
    #    pipe.model = ipex.llm.optimize(pipe.model, device=device, dtype=torch.float32, inplace=True)
    #else:
    #    torch.cuda.tunable.enable(val=True)
    #    pipe.model = torch.compile(pipe.model, mode="max-autotune", backend="inductor")

    print("Searching for WEBP files...")
    file_list = glob.glob('./**/*.webp')
    epoch_len = len(file_list)
    image_backend = ImageBackend(file_list)
    save_backend = SaveAestheticBackend()

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
                image, image_path = image_backend.get_images()[0]
                model_out = pipe(images=[image])[0]
                image.close()
                if model_out[0]["label"] == "hq":
                    prediction = model_out[0]["score"]
                elif model_out[1]["label"] == "hq":
                    prediction = model_out[1]["score"]
                aesthetic_tag = get_aesthetic_tag(prediction)
                save_backend.save(aesthetic_tag, os.path.splitext(image_path)[0]+".txt")
            except Exception as e:
                os.makedirs("errors", exist_ok=True)
                error_file = open("errors/errors.txt", 'a')
                error_file.write(f"ERROR: {image_path} MESSAGE: {e} \n")
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

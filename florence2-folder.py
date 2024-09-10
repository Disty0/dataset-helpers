#!/usr/bin/env python

import os
import gc
import glob
import time
import torch
try:
    import intel_extension_for_pytorch as ipex
except Exception:
    pass
from PIL import Image
from queue import Queue
from transformers import AutoModelForCausalLM, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

image_ext = ".webp"
prompt = "<DETAILED_CAPTION>"
model_id = "MiaoshouAI/Florence-2-base-PromptGen-v1.5"
device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"
dtype = torch.float16 if "xpu" not in device else torch.bfloat16
use_flash_atten = "xpu" not in device
steps_after_gc = 0


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

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=dtype,
    attn_implementation="flash_attention_2" if use_flash_atten else None,
).to(device, dtype=dtype).eval()
model.requires_grad_(False)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

if "xpu" in device:
    model = ipex.optimize(model, dtype=dtype, inplace=True, weights_prepack=False)
#else:
#    torch.cuda.tunable.enable(val=True)
#    model = torch.compile(model, mode="max-autotune", backend="inductor")


letter_list = [
    'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
    'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'
]

caption_list = [
    ["  ", " "],
    [" -inspired", " stylization inspired"],
    [" -focused", " art style focused"],
    [" -oriented", " art style oriented"],
    [" -themed", " stylized"],
    [" -style", " stylized"],
    [" -like", " stylized"],
    [" -related", " context related"],
    [" ish", " stylized"],
    [" ish", " stylized"],
    [" a or ", " a "],
    [" a d ", " a "],
    [" d ", " "],
    [" in a, ", ", "],
    [" a illustration", " an illustration"],
    [" and illustration", " illustration"],
    [" stylized stylization inspired" , " stylized"],
]

filler_caption_list = [
    ["the image description suggests ", ""],
    ["the image appears to depict ", ""],
    ["the image seems to capture ", ""],
    ["the image appears to show ", ""],
    ["the illustration depicts ", "an illustration of "],
    ["the image appears to be ", ""],
    ["the provided describes ", ""],
    ["the image demonstrates ", ""],
    ["the image portrayal is ", ""],
    ["the image comprises of ", ""],
    ["the image illustrates ", "an illustration of "],
    ["the image consists of ", ""],
    ["the image seems to be ", ""],
    ["the image represents ", ""],
    ["the image showcases ", ""],
    ["the image comprises ", ""],
    ["the image captures ", ""],
    ["the image contains ", ""],
    ["the image displays ", ""],
    ["the image exhibits ", ""],
    ["the image features ", ""],
    ["the image portrays ", ""],
    ["the image provides ", ""],
    ["the image presents ", ""],
    ["the image includes ", ""],
    ["the image conveys ", ""],
    ["the image depicts ", ""],
    ["the image depict ", ""],
    ["the image poses ", ""],
    ["the image shows ", ""],
    ["the image show ", ""],
    ["in this image, ", ""],
    ["in this image ", ""],
    ["in the image, ", ""],
    ["in the image ", ""],
    ["the image of ", ""],
    ["the image is ", ""],
    ["this is ", ""],
    ["a d ", "a "],
    ["as the image is described as ", ""],
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
    global caption_list, filler_caption_list
    caption = caption.lower().replace("from anime", "").replace("anime-style", "").replace("an anime style", "a").replace("anime style", "").replace("an anime image", "an image").replace("an anime", "a").replace("anime", "")
    caption = caption.replace("an animated character", "a character").replace("animated", "").replace("manga girl", "girl").replace("manga male", "male"). replace("manga character","character")
    caption = caption.replace("cartoon-style", "").replace("cartoon style", "").replace("a cartoon illustration", "an illustration").replace("a cartoon", "a").replace("cartoon", "")
    caption = caption.replace(" an character", " a character").replace(" an  girl ", " a girl ").replace("\n", " ").replace("  ", " ")
    caption = cleanup_whitespace(caption)
    if caption[:9] == "an  girl ":
        caption = "a girl " + caption[9:]
    caption = cleanup_whitespace(caption)
    for old_tag, new_tag in caption_list:
        caption = caption.replace(old_tag, new_tag)
    caption = cleanup_whitespace(caption)
    for tag, new_tag in filler_caption_list:
        tag_len = len(tag)
        if caption[:tag_len] == tag:
            caption = new_tag + caption[tag_len:]
    caption = cleanup_whitespace(caption)
    for letter in letter_list:
        caption = caption.replace(f" an {letter}", f" a {letter}")
    caption = cleanup_whitespace(caption)
    for letter in letter_list:
        if caption[:4] == f"an {letter}":
            caption = "a" + caption[2:]
            break
    caption = cleanup_whitespace(caption)
    return caption


class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGBA")
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        width, height = image.size
        image.close()
        return image_path, inputs["input_ids"][0], inputs["pixel_values"][0], width, height


class SaveCaptionBackend():
    def __init__(self, max_save_workers=2):
        self.keep_saving = True
        self.save_queue = Queue()
        self.save_thread = ThreadPoolExecutor(max_workers=max_save_workers)
        for _ in range(max_save_workers):
            self.save_thread.submit(self.save_thread_func)


    def save(self, generated_text, image_paths, widths, heights):
        self.save_queue.put([generated_text, image_paths, widths, heights])


    def save_thread_func(self):
        while self.keep_saving:
            if not self.save_queue.empty():
                generated_text, image_paths, widths, heights = self.save_queue.get()
                for i in range(len(image_paths)):
                    prediction = processor.post_process_generation(generated_text[i], task=prompt, image_size=(int(widths[i]), int(heights[i])))[prompt]
                    self.save_to_file(prediction, os.path.splitext(image_paths[i])[0]+".txt")
            else:
                time.sleep(0.1)
        print("Stopping the save backend threads")
        return


    def save_to_file(self, data, path):
        caption_file = open(path, "w")
        caption_file.write(cleanup_caption(data))
        caption_file.close()


if __name__ == '__main__':
    print(f"Searching for {image_ext} files...")
    file_list = glob.glob(f'./**/*{image_ext}')

    image_dataset = ImageDataset(file_list)
    train_dataloader = DataLoader(dataset=image_dataset, batch_size=32, shuffle=False, num_workers=8, prefetch_factor=4)
    save_backend = SaveCaptionBackend()

    with torch.no_grad():
        for image_paths, input_ids, pixel_values, widths, heights in tqdm(train_dataloader):
            try:
                generated_ids = model.generate(
                    input_ids=input_ids.to(device),
                    pixel_values=pixel_values.to(device, dtype=dtype),
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=3
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                save_backend.save(generated_text, image_paths, widths, heights)
            except Exception as e:
                os.makedirs("errors", exist_ok=True)
                error_file = open("errors/errors.txt", 'a')
                error_file.write(f"ERROR: {image_paths} MESSAGE: {e} \n")
                error_file.close()
            steps_after_gc = steps_after_gc + 1
            if steps_after_gc >= 256:
                getattr(torch, torch.device(device).type).synchronize()
                getattr(torch, torch.device(device).type).empty_cache()
                gc.collect()
                steps_after_gc = 0

    while not save_backend.save_queue.empty():
        print(f"Waiting for the remaining writes: {save_backend.save_queue.qsize()}")
        time.sleep(1)
    save_backend.keep_saving = False
    save_backend.save_thread.shutdown(wait=True)
    del save_backend

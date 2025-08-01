#!/usr/bin/env python

import os
import math
import json
import imagesize
from glob import glob
from tqdm import tqdm

try:
    import pillow_jxl # noqa: F401
except Exception:
    pass
from PIL import Image # noqa: E402

from typing import List, Tuple

remove_files = False
resize_pixel_art = False
img_ext_list = ("jpg", "png", "webp", "jpeg", "jxl")
Image.MAX_IMAGE_PIXELS = 999999999 # 178956970


class JXLBitstream:
    """
    A stream of bits with methods for easy handling.
    """

    def __init__(self, file, offset: int = 0, offsets: List[List[int]] = None):
        self.shift = 0
        self.bitstream = bytearray()
        self.file = file
        self.offset = offset
        self.offsets = offsets
        if self.offsets:
            self.offset = self.offsets[0][1]
            self.previous_data_len = 0
            self.index = 0
        self.file.seek(self.offset)

    def get_bits(self, length: int = 1) -> int:
        if self.offsets and self.shift + length > self.previous_data_len + self.offsets[self.index][2]:
            self.partial_to_read_length = length
            if self.shift < self.previous_data_len + self.offsets[self.index][2]:
                self.partial_read(0, length)
            self.bitstream.extend(self.file.read(self.partial_to_read_length))
        else:
            self.bitstream.extend(self.file.read(length))
        bitmask = 2**length - 1
        bits = (int.from_bytes(self.bitstream, "little") >> self.shift) & bitmask
        self.shift += length
        return bits

    def partial_read(self, current_length: int, length: int) -> None:
        self.previous_data_len += self.offsets[self.index][2]
        to_read_length = self.previous_data_len - (self.shift + current_length)
        self.bitstream.extend(self.file.read(to_read_length))
        current_length += to_read_length
        self.partial_to_read_length -= to_read_length
        self.index += 1
        self.file.seek(self.offsets[self.index][1])
        if self.shift + length > self.previous_data_len + self.offsets[self.index][2]:
            self.partial_read(current_length, length)


def decode_codestream(file, offset: int = 0, offsets: List[List[int]] = None) -> Tuple[int,int]:
    """
    Decodes the actual codestream.
    JXL codestream specification: http://www-internal/2022/18181-1
    """

    # Convert codestream to int within an object to get some handy methods.
    codestream = JXLBitstream(file, offset=offset, offsets=offsets)

    # Skip signature
    codestream.get_bits(16)

    # SizeHeader
    div8 = codestream.get_bits(1)
    if div8:
        height = 8 * (1 + codestream.get_bits(5))
    else:
        distribution = codestream.get_bits(2)
        match distribution:
            case 0:
                height = 1 + codestream.get_bits(9)
            case 1:
                height = 1 + codestream.get_bits(13)
            case 2:
                height = 1 + codestream.get_bits(18)
            case 3:
                height = 1 + codestream.get_bits(30)
    ratio = codestream.get_bits(3)
    if div8 and not ratio:
        width = 8 * (1 + codestream.get_bits(5))
    elif not ratio:
        distribution = codestream.get_bits(2)
        match distribution:
            case 0:
                width = 1 + codestream.get_bits(9)
            case 1:
                width = 1 + codestream.get_bits(13)
            case 2:
                width = 1 + codestream.get_bits(18)
            case 3:
                width = 1 + codestream.get_bits(30)
    else:
        match ratio:
            case 1:
                width = height
            case 2:
                width = (height * 12) // 10
            case 3:
                width = (height * 4) // 3
            case 4:
                width = (height * 3) // 2
            case 5:
                width = (height * 16) // 9
            case 6:
                width = (height * 5) // 4
            case 7:
                width = (height * 2) // 1
    return width, height


def decode_container(file) -> Tuple[int,int]:
    """
    Parses the ISOBMFF container, extracts the codestream, and decodes it.
    JXL container specification: http://www-internal/2022/18181-2
    """

    def parse_box(file, file_start: int) -> dict:
        file.seek(file_start)
        LBox = int.from_bytes(file.read(4), "big")
        XLBox = None
        if 1 < LBox <= 8:
            raise ValueError(f"Invalid LBox at byte {file_start}.")
        if LBox == 1:
            file.seek(file_start + 8)
            XLBox = int.from_bytes(file.read(8), "big")
            if XLBox <= 16:
                raise ValueError(f"Invalid XLBox at byte {file_start}.")
        if XLBox:
            header_length = 16
            box_length = XLBox
        else:
            header_length = 8
            if LBox == 0:
                box_length = os.fstat(file.fileno()).st_size - file_start
            else:
                box_length = LBox
        file.seek(file_start + 4)
        box_type = file.read(4)
        file.seek(file_start)
        return {
            "length": box_length,
            "type": box_type,
            "offset": header_length,
        }

    file.seek(0)
    # Reject files missing required boxes. These two boxes are required to be at
    # the start and contain no values, so we can manually check there presence.
    # Signature box. (Redundant as has already been checked.)
    if file.read(12) != bytes.fromhex("0000000C 4A584C20 0D0A870A"):
        raise ValueError("Invalid signature box.")
    # File Type box.
    if file.read(20) != bytes.fromhex(
        "00000014 66747970 6A786C20 00000000 6A786C20"
    ):
        raise ValueError("Invalid file type box.")

    offset = 0
    offsets = []
    data_offset_not_found = True
    container_pointer = 32
    file_size = os.fstat(file.fileno()).st_size
    while data_offset_not_found:
        box = parse_box(file, container_pointer)
        match box["type"]:
            case b"jxlc":
                offset = container_pointer + box["offset"]
                data_offset_not_found = False
            case b"jxlp":
                file.seek(container_pointer + box["offset"])
                index = int.from_bytes(file.read(4), "big")
                offsets.append([index, container_pointer + box["offset"] + 4, box["length"] - box["offset"] - 4])
        container_pointer += box["length"]
        if container_pointer >= file_size:
            data_offset_not_found = False

    if offsets:
        offsets.sort(key=lambda i: i[0])
    file.seek(0)

    return decode_codestream(file, offset=offset, offsets=offsets)


def get_jxl_size(path: str) -> Tuple[int,int]:
    with open(path, "rb") as file:
        if file.read(2) == bytes.fromhex("FF0A"):
            return decode_codestream(file)
        return decode_container(file)


print(f"Searching for {img_ext_list} files...")
file_list = []
for ext in img_ext_list:
    file_list.extend(glob(f"**/*.{ext}"))

os.makedirs("out", exist_ok=True)
small_res_file = open("out/small_res_check.txt", 'a')
small_res_pixel_art_file = open("out/small_res_pixel_art.txt", 'a')

for image_path in tqdm(file_list):
    try:
        if os.path.splitext(image_path)[-1] == ".jxl":
            width, height = get_jxl_size(image_path)
        else:
            width, height = imagesize.get(image_path)
        image_size = width * height
        if image_size < 768000: # 600x1280
            json_path = os.path.splitext(image_path)[0] + ".json"
            with open(json_path, "r") as f:
                json_data = json.load(f)
            if "pixel_art" not in json_data["tag_string_general"].split(" ") and "pixel_art" not in json_data.get("wd_tag_string_general", "").split(" "):
                if remove_files:
                    os.remove(image_path)
                small_res_file.write(image_path+"\n")
            else:
                if resize_pixel_art:
                    image = Image.open(image_path)
                    width, height = image.size
                    image_size = width * height
                    scale = math.sqrt(image_size / 1048576)
                    new_width = int(width/scale)
                    new_height = int(height/scale)
                    image = image.convert("RGBA").resize((new_width, new_height), Image.NEAREST)
                    image.save(image_path, lossless=True)
                    image.close()
                small_res_pixel_art_file.write(image_path+"\n")
    except Exception as e:
        os.makedirs("errors", exist_ok=True)
        error_file = open("errors/errors.txt", 'a')
        error_file.write(f"ERROR: {image_path} MESSAGE: {e}\n")
        error_file.close()
small_res_file.close()

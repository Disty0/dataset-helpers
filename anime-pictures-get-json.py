#!/usr/bin/env python

import os
import json
import time
import argparse
import requests
from tqdm import tqdm


def get_json_data(session, id):
    raw_data = json.loads(session.get(f"https://api.anime-pictures.net/api/v3/posts/{id}").content)
    if raw_data.get("redirect", None) is not None:
        return raw_data
    json_data = raw_data.pop("post")
    json_data["image_width"] = json_data.pop("width")
    json_data["image_height"] = json_data.pop("height")
    json_data["file_size"] = json_data.pop("size")
    json_data["anime_pictures_file_url"] = raw_data.pop("file_url")
    json_data["anime_pictures_score"] = json_data.pop("score")
    json_data["score"] = json_data.pop("score_number")
    json_data["file_ext"] = json_data["anime_pictures_file_url"].rsplit(".", maxsplit=1)[-1]
    json_data["file_url"] = "https://oimages.anime-pictures.net/" + json_data["md5"][:3] + "/" + json_data["md5"] + "." + json_data["file_ext"]
    for key, value in raw_data.items():
        json_data["anime_pictures_" + key] = value
    
    tag_string_general = []
    tag_string_artist = []
    tag_string_copyright = []
    tag_string_character = []
    tag_string_meta = []

    for tag_dict in json_data["anime_pictures_tags"]:
        if tag_dict["relation"]["removetime"] is None:
            tag = tag_dict["tag"]["tag"].replace(" ", "_")
            tag_type = tag_dict["tag"]["type"]
            if tag_type == 1:
                tag_string_character.append(tag)
            elif tag_type == 4:
                tag_string_artist.append(tag)
            elif tag_type in {2, 7}:
                if tag == "girl":
                    tag = "1girl"
                elif len(tag) > 5 and tag.endswith("_girls") and tag[:2] in {"2_", "3_", "4_", "5_", "6_", "7_", "8_", "9_", "10", "11", "12"}:
                    tag = tag.replace("_", "")
                    if "1girl" in tag_string_general:
                        tag_string_general.pop(tag_string_general.index("1girl"))
                tag_string_general.append(tag)
            elif tag_type in {3,5,6}:
                tag_string_copyright.append(tag)
            else:
                tag_string_meta.append(tag)

    json_data["tag_string_general"] = " ".join(tag_string_general)
    json_data["tag_string_artist"] = " ".join(tag_string_artist)
    json_data["tag_string_copyright"] = " ".join(tag_string_copyright)
    json_data["tag_string_character"] = " ".join(tag_string_character)
    json_data["tag_string_meta"] = " ".join(tag_string_meta)

    return json_data


parser = argparse.ArgumentParser(description='Get json responses from anime-pictures.net')
parser.add_argument('start', type=int)
parser.add_argument('end', type=int)
parser.add_argument('--jwt', default='', type=str)
args = parser.parse_args()

session = requests.Session() 
session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0"
session.cookies.set("kira", str(args.end + 1), domain='.anime-pictures.net')

# anime_pictures_jwt is used for login. Get anime_pictures_jwt string from 'F12 -> storage' in your browser
if len(args.jwt) > 0:
    session.cookies.set("anime_pictures_jwt", args.jwt, domain='.anime-pictures.net')
elif len(os.environ.get('ANIME_PICTURES_JWT', '')) > 0:
    session.cookies.set("anime_pictures_jwt", os.environ.get('ANIME_PICTURES_JWT'), domain='.anime-pictures.net')


for id in tqdm(range(args.start, args.end)):
    folder = str(int(id / 10000))
    if True:
        json_path = os.path.join(folder, f"{id}.json")
        jpg_path = None

        if os.path.exists(json_path) and os.path.getsize(json_path) != 0:
            continue
        else:
            try:
                time.sleep(0.25)
                image_data = get_json_data(session, id)
                os.makedirs(folder, exist_ok=True)
                with open(json_path, "w") as f:
                    json.dump(image_data, f)
            except Exception as e:
                os.makedirs("errors", exist_ok=True)
                error_file = open(f"errors/errors_json{args.start}.txt", 'a')
                error_file.write(f"ERROR: {id} MESSAGE: {str(e)}\n")
                error_file.close()
                continue

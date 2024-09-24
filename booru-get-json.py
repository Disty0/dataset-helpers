#!/usr/bin/env python

import os
import json
import time
import argparse
import pybooru
from tqdm import tqdm

pybooru.resources.SITE_LIST["shima"] = {"url": "https://shima.donmai.us/"} # get around the government censorship
client = pybooru.Danbooru('shima')

parser = argparse.ArgumentParser(description='Get json files from danbooru')
parser.add_argument('start', type=int)
parser.add_argument('end', type=int)
args = parser.parse_args()


for id in tqdm(range(args.start, args.end)):
    folder = str(int(id / 10000))
    json_path = os.path.join(folder, f"{id}.json")
    if not (os.path.exists(json_path) and os.path.getsize(json_path) != 0):
        try:
            time.sleep(0.25)
            json_data = client.post_show(id)
            os.makedirs(folder, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(json_data, f)
        except Exception as e:
            str_e = str(e)
            if not ("In _request: 404 - Not Found" in str_e and str_e.endswith(".json")):
                os.makedirs("errors", exist_ok=True)
                error_file = open(f"errors/errors_json{args.start}.txt", 'a')
                error_file.write(f"ERROR: {id} MESSAGE: {str_e}\n")
                error_file.close()

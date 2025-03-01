import json
import logging
import os
from typing import List
import ipdb
import lmdb
import numpy as np
import requests
from PIL import Image
from pathlib import Path

import sys

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from apis.global_vars import CLIP_CACHE_FILE, CLIP_URL
from cache.cache_utils import get_from_cache, save_to_cache
from cache import cache_utils

HITS = 0
MISSES = 0

if not os.path.exists(CLIP_CACHE_FILE):
    os.makedirs(CLIP_CACHE_FILE)

clip_cache = lmdb.open(CLIP_CACHE_FILE, map_size=int(1e11))


def get_embeddings(inputs: List[str], model: str, modality: str, raise_on_clip_fail=True) -> np.ndarray:
    global HITS, MISSES
    print(f"\rCLIP server cache. Hits: {HITS}. Misses: {MISSES}", end="")
    input_to_embeddings = {}
    for inp in inputs:
        key = json.dumps([inp, model])
        cached_value = get_from_cache(key, clip_cache)
        if cached_value is not None:
            logging.debug(f"CLIP Cache Hit")
            HITS += 1 
            input_to_embeddings[inp] = json.loads(cached_value)
        else:
            MISSES += 1

    uncached_inputs = [inp for inp in inputs if inp not in input_to_embeddings]

    if len(uncached_inputs) > 0:
        try:
            response = requests.post(CLIP_URL,
                                     data={
                                         modality: json.dumps(uncached_inputs)
                                     }).json()
            for inp, embedding in zip(uncached_inputs, response["embeddings"]):
                input_to_embeddings[inp] = embedding
                key = json.dumps([inp, model])
                save_to_cache(key, json.dumps(embedding), clip_cache)
        except Exception as e:
            if raise_on_clip_fail:
                raise e

            logging.error(f"CLIP Error: {e}")
            for inp in uncached_inputs:
                input_to_embeddings[inp] = None

    input_embeddings = [input_to_embeddings[inp] for inp in inputs]
    return np.array(input_embeddings)


def get_embeddings_video(video: np.ndarray, model: str) -> np.ndarray:
    """ 
    A wrapper around `get_embeddings` for embedding a whole video. 

    The `get_embeddings` for images takes a list of filenames. 
    Instead, this method takes a video numpy array, checks the cache for the 
    answer. If not cached, save the frames to /tmp. 

    Note: the filenames have a hash key in them. That's important because the 
    `get_embeddings` func uses the filename as a hash key
    """
    assert video.ndim == 4 and video.shape[-1] == 3
    hash_array = cache_utils.hash_array(video)
    key = json.dumps([hash_array, model])
    cached_value = get_from_cache(key, clip_cache)

    if cached_value is not None:
        logging.debug(f"CLIP Cache Hit")
        embeddings = np.array(json.loads(cached_value))
        return embeddings

    # if not cached, then save the frames as images to unique fnames
    Path("tmp").mkdir(exist_ok=True)
    fnames = [f"tmp/arr_{hash_array}_frame_{i}.png" for i in range(len(video))]
    _ = [Image.fromarray(frame).save(fname) for (frame,fname) in zip(video, fnames)]

    # call the old embedding function
    embeddings = get_embeddings(fnames, model, "image")

    # save to cache
    save_to_cache(key, json.dumps(embeddings.tolist()), clip_cache)
    
    return embeddings


if __name__ == "__main__":

    ipdb.set_trace()
    embeddings = get_embeddings(["haha", "hello world"], "ViT-bigG-14", "text")
    print(embeddings)

    ipdb.set_trace()
    embeddings = get_embeddings(
        ["data/teaser.png"],
        "ViT-bigG-14",
        "image",
    )
    print(embeddings)

"""
python -m ipdb apis/llavavideo_api.py
"""

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
import ipdb
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
from tqdm import trange
import torch
import sys
import warnings
import lmdb
import json
from threading import Lock
import numpy as np

warnings.filterwarnings("ignore")

from cache import cache_utils

cache_llavavideo = lmdb.open("cache/cache_llavavideo", map_size=int(1e12))
cache_lock = Lock()


def load_model(pretrained="lmms-lab/LLaVA-Video-7B-Qwen2", model_name="llava_qwen"):

    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map
    )
    model.eval()
    model_dict = dict(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        pretrained=pretrained,
        model_name=model_name,
        max_length=max_length,
    )
    return model_dict


def call_llavavideo(
    # args for setting the `messages` param
    model_dict: dict,
    text: str,
    video: list[np.ndarray] = None,
    system_prompt: str = None,
    seed: int = 0,
    # args for caching behaviour
    cache: bool = True,
    overwrite_cache: bool = False,
    # temp directory where to store covnerted arrays in order to submit to gemini
    temp_dir: str = "VideoCache",
):
    """ """
    assert len(video) == 1
    assert "<|im_start|>" in text, "llava-video processing"

    model = model_dict["model"]
    image_processor = model_dict["image_processor"]
    tokenizer = model_dict["tokenizer"]

    if cache:
        # warning that cache kwargs should depend on tokenizer and processor kwargs too
        cache_kwargs = dict(
            text=text,
            imgs_hash_key=cache_utils.hash_array(video[0].numpy()),
            pretrained=model_dict["pretrained"],
            model_name=model_dict["model_name"],
        )
        cache_key = json.dumps(cache_kwargs, sort_keys=True)
        with cache_lock:
            msg = cache_utils.get_from_cache(cache_key, cache_llavavideo)
        if msg is not None and not overwrite_cache:
            return msg, None

    input_ids = (
        tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    video_processed = (
        image_processor.preprocess(video[0].to(torch.float16), return_tensors="pt")[
            "pixel_values"
        ]
        .cuda()
        .half()
    )
    video_processed = [video_processed.to(torch.bfloat16)]
    cont = model.generate(
        input_ids,
        images=video_processed,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    if cache:
        with cache_lock:
            cache_utils.save_to_cache(cache_key, text_outputs, cache_llavavideo)

    return text_outputs, None


def call_llavavideo_batch(
    batch_prompts_text,
    batch_prompts_video,
    seeds=None,
    model="lmms-lab/LLaVA-Video-7B-Qwen2",
    debug=None,
    json_mode=False,
):
    model_dict = load_model(pretrained="lmms-lab/LLaVA-Video-7B-Qwen2")

    n = len(batch_prompts_text)
    msgs = []
    responses = []
    assert n == len(batch_prompts_video)

    for i in trange(n):
        text = batch_prompts_text[i]
        video = batch_prompts_video[i]
        msg, response = call_llavavideo(model_dict=model_dict, text=text, video=video)
        msgs.append(msg)
        responses.append(response)

    return msgs, responses


if __name__ == "__main__":
    from data import load_viddiff_dataset as lvd

    dataset = lvd.load_viddiff_dataset(["easy"])
    videos = lvd.load_all_videos(dataset)

    # sample one video
    idx = 0
    video0, video1 = videos[0][idx], videos[1][idx]
    videos = [video0["video"], video1["video"]]
    fpss = [video0["fps"], video1["fps"]]

    # load a model
    model_dict = load_model()
    text = "compare these videos"

    # call
    res = call_llavavideo(model_dict, text, videos, fpss)
    ipdb.set_trace()
    pass

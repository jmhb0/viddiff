"""
Functions for calling Gemini api
Needs to have set GOOGLE_API_KEY.
Models: https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
"""

import os
import sys
import ipdb
import time
import lmdb
import json
import numpy as np
from typing import List
from threading import Lock
import cv2
import random
import numpy as np
import google.generativeai as genai
from tqdm import trange
import logging
import concurrent.futures
from google.generativeai.types import HarmCategory, HarmBlockThreshold

sys.path.insert(0, ".")
from cache import cache_utils

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
cache_gemini = lmdb.open("cache/gemini", map_size=int(1e11))
cache_lock = Lock()


def select_random_items(input_list, N: int = 3):
    if len(input_list) < N:
        raise ValueError("The list must contain at least 3 items.")
    return random.sample(input_list, N)


def video_to_numpy_array(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Append the frame to the list of frames
        frames.append(frame)

    cap.release()

    video_array = np.array(frames)

    return video_array


def numpy_array_to_video(video_array, output_path, fps=30):
    """
    Converts a NumPy array of video frames into a video file.

    Args:
    video_array (np.ndarray): 4D NumPy array containing the video frames.
                              Shape should be (frames, height, width, channels).
    output_path (str): Path to save the output video.
    fps (int, optional): Frames per second of the output video. Default is 30.
    """
    # Extract the height, width, and number of channels from the video array
    height, width, channels = video_array[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in video_array:
        # Write each frame to the video file
        out.write(frame)

    # Release the video writer object
    out.release()


def call_gemini(
    # args for setting the `messages` param
    text: str,
    videos: List[np.ndarray] = None,
    system_prompt: str = None,
    json_mode: bool = False,
    # kwargs for client.chat.completions.create
    detail: str = "high",
    model: str = "gemini-1.5-pro-latest",
    temperature: float = 1,
    max_tokens: int = 2048,
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    seed: int = 0,
    n: int = 1,
    # args for caching behaviour
    cache: bool = True,
    overwrite_cache: bool = False,
    num_retries:
    # if json_mode=True, and not json decodable, retry this many time
    int = 10,
    # temp directory where to store covnerted arrays in order to submit to gemini
    temp_dir: str = "tmp",
    fps: int = 30,
):
    """
    models: ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
    """
    if isinstance(system_prompt, str):
        text: str = system_prompt + " " + text

    content = [{"type": "text", "text": text}]

    video_cache: list = []
    if videos is not None:
        for video in videos:
            for frame in video:
                # the cache value also needs to include the fps
                video_cache.append(cache_utils.hash_array(frame))

        content.append({"imgs_hash_key": video_cache})

    # save for caching: args to the call for client.chat.completions.create
    kwargs = dict(
        messages=content,
        response_format="None",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        fps=fps,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        n=n,
    )

    if cache:
        cache_key = json.dumps(kwargs, sort_keys=True)
        with cache_lock:
            msg = cache_utils.get_from_cache(cache_key, cache_gemini)
        if msg is not None and not overwrite_cache:
            if json_mode:
                msg = json.loads(msg)
            return msg, None

    # Gemini takes a list of things:
    content_call: list = [text]

    if videos is not None:
        assert "imgs_hash_key" in content[-1].keys()
        # Convert NumPy array back to video
        for video in videos:
            hash_list: list[str] = select_random_items(
                content[-1]["imgs_hash_key"], N=3
            )
            joined_hashes: str = "".join(hash_list)

            os.makedirs(temp_dir, exist_ok=True)
            video_file_name: str = os.path.join(temp_dir, f"{joined_hashes}.mp4")

            numpy_array_to_video(video, output_path=video_file_name, fps=fps)

            video_file = genai.upload_file(path=video_file_name)

            while video_file.state.name == "PROCESSING":
                # logging.info('Uploading video to gemini api')
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
                content_call.append(video_file)

            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)

    gemini_model = genai.GenerativeModel(model_name=model)
    response = gemini_model.generate_content(
        content_call,
        request_options={"timeout": 600},
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
    )

    prompt_tokens = response.usage_metadata.prompt_token_count
    completion_tokens = response.usage_metadata.candidates_token_count
    msg = response.text

    # save to cache if enabled
    if cache:
        with cache_lock:
            cache_utils.save_to_cache(cache_key, msg, cache_gemini)

    if os.path.exists(video_file_name):
        os.remove(video_file_name)

    price = compute_api_call_cost(prompt_tokens, completion_tokens, model=model)

    return msg, response


def call_gemini_batch(texts, videos, seeds, debug=None, **kwargs):
    """
    no multithreading for now
    """
    n = len(texts)
    msgs = []
    responses = []

    # code for doing it in serial
    logging.info(f"Running Gemini NOT in parallel")
    for i in trange(n):
        # if debug:
        #     print(debug[i])
        msg, res = call_gemini(texts[i], videos[i], seeds[i], **kwargs)
        msgs.append(msg)
        responses.append(res)

    return msgs, responses


def compute_api_call_cost(
    prompt_tokens: int, completion_tokens: int, model="models/gemini-1.5-pro"
):
    """
    Warning: prices need to be manually updated from
    https://openai.com/api/pricing/
    """
    prices_per_million_input = {
        "models/gemini-1.5-pro": 3.50,
        "models/gemini-1.5-flash": 0.075,
        # "gpt-4o-mini": 0.15,
        # "gpt-4o": 5,
        # "gpt-4-turbo": 10,
        # "gpt-4": 30,
        # "gpt-3.5-turbo": 0.5
    }
    prices_per_million_output = {
        "models/gemini-1.5-pro": 10.50,
        "models/gemini-1.5-flash": 0.30,
        # "gpt-4o-mini": 0.075,
        # "gpt-4o": 15,
        # "gpt-4-turbo": 30,
        # "gpt-4": 60,
        # "gpt-3.5-turbo": 1.5
    }
    key = model

    price = (
        prompt_tokens * prices_per_million_input[key]
        + completion_tokens * prices_per_million_output[key]
    )
    price = price / 1e6

    return price


if __name__ == "__main__":
    text0 = "Explain the differences between this two videos. Use JSON {'answer':'...'}"

    # Convert video to NumPy array
    video_path_1 = "data/src_fitnessaqa/Squat/Labeled_Dataset/videos/37640_3.mp4"
    video_path_2 = "data/src_fitnessaqa/Squat/Labeled_Dataset/videos/48331_1.mp4"

    video_array_1 = video_to_numpy_array(video_path_1)
    video_array_2 = video_to_numpy_array(video_path_2)
    videos: list[np.array] = [video_array_1, video_array_2]

    msg, response = call_gemini(
        text=text0,
        model="models/gemini-1.5-flash",
        cache=False,
        videos=videos,
        json_mode=False,
    )
    ipdb.set_trace()
    pass

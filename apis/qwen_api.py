import os
import sys
import json
import ipdb
import lmdb
import torch
import numpy as np
from tqdm import trange
from threading import Lock
from qwen_vl_utils import process_vision_info
import cv2
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

sys.path.insert(0, "..")
sys.path.insert(0, ".")
from cache import cache_utils
from apis.gemini_api import (
    select_random_items,
    video_to_numpy_array,
    numpy_array_to_video,
)

cache_qwen = lmdb.open("cache/cache_qwen", map_size=int(1e11))
cache_lock = Lock()


def get_qwen_model(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    verbose: bool = True,
    device: str = "auto",
) -> dict:
    """
    Load the Qwen model and processor for conditional generation.

    Parameters
    ----------
    model_name : str, optional
        The name or path of the model to load. Default is "Qwen/Qwen2-VL-72B-Instruct".
    torch_dtype : torch.dtype, optional
        The desired data type for the model. Default is torch.bfloat16.
    verbose : bool, optional
        If True, prints the loaded model name. Default is True.
    device : str, optional
        The device on which to load the model. It can be "cpu", "cuda", or "auto"
        to automatically choose the best device. Default is "auto".

    Returns
    -------
    dict
        A dictionary containing the loaded model, processor, model name, and device:
        - 'model': The Qwen model loaded for conditional generation.
        - 'model_name': The name of the loaded model.
        - 'processor': The processor used for preparing input data.
        - 'device': The device used for model loading.

    Examples
    --------
    >>> model_dict = get_qwen_model(model_name="Qwen/Qwen2-VL-72B-Instruct", device="cuda")
    Loaded: Qwen/Qwen2-VL-72B-Instruct
    >>> model = model_dict['model']
    >>> processor = model_dict['processor']
    """

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        device_map=device,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    model_dict = {
        "model": model,
        "model_name": model_name,
        "processor": processor,
        "device": device,
    }

    if verbose:
        print(f"Loaded: {model_name}")

    return model_dict


def call_qwen2V(
    # args for setting the `messages` param
    model_dict: dict,
    text: str,
    max_pixles: int = 360 * 420,
    videos: list[np.ndarray] = None,
    system_prompt: str = None,
    json_mode: bool = False,
    # kwargs for client.chat.completions.creat
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
    temp_dir: str = "VideoCache",
    fps: int = 30,
):
    """
    Generate responses using the Qwen model with optional video input and caching.

    Parameters
    ----------
    model_dict : dict
        Dictionary containing the model, processor, and device information.
    text : str
        Input text to be used for the generation task.
    max_pixles : int, optional
        Maximum number of pixels for the video, used to limit the size of the video. Default is 360 * 420.
    videos : list of np.ndarray, optional
        List of videos as NumPy arrays. Each video is an array of frames (e.g., [frames, height, width, channels]).
        Default is None.
    system_prompt : str, optional
        Optional prompt to prepend to the input text. Default is None.
    json_mode : bool, optional
        If True, parses and returns JSON data. Default is False.
    temperature : float, optional
        Sampling temperature for text generation. Default is 1.
    max_tokens : int, optional
        Maximum number of tokens to generate. Default is 2048.
    top_p : float, optional
        Probability threshold for nucleus sampling. Default is 1.
    frequency_penalty : float, optional
        Penalty for repeated tokens. Default is 0.
    presence_penalty : float, optional
        Penalty for new tokens based on presence in the input. Default is 0.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    n : int, optional
        Number of responses to generate. Default is 1.
    cache : bool, optional
        If True, enables caching for faster subsequent calls. Default is True.
    overwrite_cache : bool, optional
        If True, overwrites the cache even if cached data exists. Default is False.
    num_retries : int, optional
        Number of times to retry if the response is not JSON-decodable and `json_mode` is True. Default is 10.
    temp_dir : str, optional
        Temporary directory for storing converted videos before submission. Default is "VideoCache".
    fps : int, optional
        Frames per second for the output video. Default is 30.

    Returns
    -------
    tuple
        - msg (str or list): The generated message or JSON-parsed response.
        - response (dict): A dictionary with token usage information:
            - 'prompt_tokens': The number of tokens in the prompt.
            - 'completion_tokens': The number of tokens in the completion.

    Examples
    --------
    >>> model_dict = get_qwen_model()
    >>> videos = [np.random.rand(10, 360, 420, 3)]  # A sample video as a list of frames
    >>> message, response = call_qwen2V(
    ...     model_dict=model_dict,
    ...     text="Describe the content of the video.",
    ...     videos=videos
    ... )
    >>> print(message)
    ('Video description', {'prompt_tokens': 20, 'completion_tokens': 150})
    """

    if isinstance(system_prompt, str):
        text: str = system_prompt + " " + text
    content = [{"type": "text", "text": text}]

    # Cashing Frames
    video_cahce: list = []
    if videos is not None:
        for video in videos:
            for frame in video:
                video_cahce.append(cache_utils.hash_array(frame))
        content.append({"imgs_hash_key": video_cahce})

    #  Just saving for caching :  arguments to the call for client.chat.completions.create
    kwargs = dict(
        messages=content,
        response_format="None",
        model=model_dict["model_name"],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
        n=n,
    )

    if cache:
        cache_key = json.dumps(kwargs, sort_keys=True)
        with cache_lock:
            msg = cache_utils.get_from_cache(cache_key, cache_qwen)

        if msg is not None and not overwrite_cache:
            return msg, None

    if videos is not None:
        assert "imgs_hash_key" in content[-1].keys()
        # Convert NumPy array back to video
        video_keys = content.pop(-1)
        for i, video in enumerate(videos):
            hash_list: list[str] = select_random_items(video_keys["imgs_hash_key"], N=3)
            joined_hashes: str = "".join(hash_list)

            os.makedirs(temp_dir, exist_ok=True)
            video_file_name: str = os.path.join(temp_dir, f"{joined_hashes}.mp4")

            numpy_array_to_video(video, output_path=video_file_name, fps=fps)

            content.append({"type": "text", "text": f"Video {i}:"})
            content.append(
                {
                    "type": "video",
                    "video": video_file_name,
                    "max_pixels": max_pixles,
                    "fps": fps,
                }
            )

    messages = [{"role": "user", "content": content}]

    print("*" * 80)
    print(content)

    # Preparation for inference
    text = model_dict["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = model_dict["processor"](
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model_dict["model"].generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    msg = model_dict["processor"].batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    prompt_tokens = len(generated_ids_trimmed[0])
    completion_tokens = len(generated_ids_trimmed[0])

    assert len(msg) == 1
    msg = msg[0]

    # save to cache if enabled
    if cache:
        with cache_lock:
            cache_utils.save_to_cache(cache_key, msg, cache_qwen)

    response = dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    if os.path.exists(video_file_name):
        os.remove(video_file_name)

    return msg, response


def call_qwen_batch(
    batch_prompts_text,
    batch_prompts_video,
    seeds=None,
    model="Qwen/Qwen2-VL-7B-Instruct",
    debug=None,
    json_mode=False,
):
    model_dict: dict = get_qwen_model(
        model_name=model, torch_dtype=torch.bfloat16, device="auto"
    )

    n = len(batch_prompts_text)
    msgs = []
    responses = []
    assert n == len(batch_prompts_video)

    for i in trange(n):
        text = batch_prompts_text[i]
        video = batch_prompts_video[i]
        if "json" not in text.lower():
            raise "Qwen expects json mode"
        msg, response = call_qwen2V(
            model_dict=model_dict, text=text, cache=True, videos=video, json_mode=False
        )
        msgs.append(msg)
        responses.append(response)

    return msgs, responses


# basic testing
if __name__ == "__main__":
    # sys.path.insert(0, "..")
    model_dict: dict = get_qwen_model(
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device="auto",
    )

    instruction = """
    Explain the differences between the video 1 and video 2, be concise. 
    Put your answer in json like {"reasons" : ["...","...", ...]}}
    """

    # Convert video to NumPy array
    video_path_1 = "data/src_fitnessaqa/Squat/Labeled_Dataset/videos/37640_3.mp4"
    video_path_2 = "data/src_fitnessaqa/Squat/Labeled_Dataset/videos/48331_1.mp4"

    video_array_1 = video_to_numpy_array(video_path_1)
    video_array_2 = video_to_numpy_array(video_path_2)

    print(f"Video array shape: {video_array_1.shape}")
    print(f"Video array shape: {video_array_2.shape}")

    videos: list[np.array] = [video_array_1, video_array_2]

    msg, response = call_qwen2V(
        model_dict=model_dict,
        text=instruction,
        cache=True,
        videos=videos,
        json_mode=False,
    )

    ipdb.set_trace()

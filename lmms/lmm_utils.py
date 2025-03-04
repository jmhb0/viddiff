import ipdb
from datasets import Dataset
import numpy as np
import logging
from omegaconf.basecontainer import BaseContainer
import numpy as np
import sys
import json
import pandas as pd
import re
import torchvision.transforms as T
import torch
import copy
import lmdb

sys.path.insert(0, ".")

from apis import openai_api
from apis import gemini_api
import data.load_viddiff_dataset as lvd
import lmms.lmm_prompts as lp


def make_text_prompts(
    dataset: Dataset,
    videos: tuple,
    n_differences: list,
    eval_mode: int,
    args_lmm: BaseContainer,
):

    batch_prompts_text, patch_prompt_videos = [], []
    for i, row in enumerate(dataset):

        if eval_mode == "closed":
            keys_gt = {k for k, v in row["differences_gt"].items() if v in ("a", "b")}
            differences_annotated = row["differences_annotated"]
            differences_annotated = {
                k: v["description"]
                for (k, v) in differences_annotated.items()
                if k in keys_gt
            }

            n_differences = [None] * len(dataset)

        elif eval_mode == "open":
            differences_annotated = None

        else:
            raise ValueError()

        prompt_text, prompt_video = make_prompt(
            row["action_description"],
            videos[0][i],
            videos[1][i],
            eval_mode,
            args_lmm,
            n_differences[i],
            domain=row["domain"],
            differences_annotated=differences_annotated,
            model=args_lmm.model,
        )
        batch_prompts_text.append(prompt_text)
        patch_prompt_videos.append(prompt_video)

    return batch_prompts_text, patch_prompt_videos


def make_prompt(
    action_description: str,
    video0: dict,
    video1: dict,
    eval_mode: int,
    args_lmm: BaseContainer,
    n_difference: int = None,
    domain: str = None,
    differences_annotated: dict = None,
    model: str = None,
):
    """
    create the text and video prompts
    The possible representations are: {'frames','video', 'first_frame'}
    """

    if eval_mode == "closed":
        prompt_text = lp.prompt_template_mode_2
        prompt_text = prompt_text.replace(
            "{differences_annotated}", json.dumps(differences_annotated, indent=2)
        )
        if "qwen" not in model.lower():
            target = {
                k: {"description": v, "prediction": "a|b"}
                for k, v in differences_annotated.items()
            }
        # deal with this exception. Qwen didn't follow this instruction well.
        else:
            target = {
                k: {"description": v, "prediction": "..."}
                for k, v in differences_annotated.items()
            }

        prompt_text = prompt_text.replace("{target_out}", json.dumps(target, indent=2))

    elif eval_mode == "open":
        prompt_text = lp.prompt_template_open
        prompt_text = prompt_text.replace("{n_differences}", str(n_difference))

    else:
        raise ValueError()

    prompt_text = prompt_text.replace("{action_description}", action_description)

    # all videos have tha subsampling step
    for video in (video0, video1):
        fps_inference = args_lmm.fps_inference[domain]

        video["video"], fps_new, subsample_time_int = lvd._subsample_video(
            video["video"], video["fps_original"], fps_inference, args_lmm.fps_warning
        )

    # handle the video representation
    if args_lmm.video_representation == "frames":

        # create the images prompt
        nframes = []
        fps_new_images = []
        prompt_videos = []

        for video in (video0, video1):
            nframes.append(len(video["video"]))
            fps_new_images.append(fps_new)
            prompt_videos += list(video["video"])

        assert fps_new_images[0] == fps_new_images[1]

        # describe the video representation
        video_rep_description = lp.video_rep_description_frames
        video_rep_description = video_rep_description.replace(
            "{vid0_nframes}", str(nframes[0])
        )
        video_rep_description = video_rep_description.replace(
            "{vid1_nframes}", str(nframes[1])
        )
        video_rep_description = video_rep_description.replace(
            "{fps}", str(fps_new_images[0])
        )

        total_frames = nframes[0] + nframes[1]
        if total_frames > args_lmm.max_imgs:
            raise ValueError(
                f"Total frames [{total_frames}] is more than the "
                "max frames set in the config lmms.max_imgs. Change the "
                "max_frames or lower the config value for lmms.fps"
            )

        prompt_text = prompt_text.replace(
            "{video_representation_description}", video_rep_description
        )

    elif args_lmm.video_representation == "video":
        video_rep_description = lp.video_rep_description_2_videos
        prompt_text = prompt_text.replace(
            "{video_representation_description}", video_rep_description
        )

        prompt_videos = [video0["video"], video1["video"]]

    elif args_lmm.video_representation == "first_frame":
        video_rep_description = lp.video_rep_description_first_frame
        prompt_text = prompt_text.replace(
            "{video_representation_description}", video_rep_description
        )

        prompt_videos = [video0["video"][0], video1["video"][0]]

    elif args_lmm.video_representation == "llavavideo":
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import (
            IMAGE_TOKEN_INDEX,
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            IGNORE_INDEX,
        )

        fps0, fps1 = video0["fps_original"], video1["fps_original"]
        nframes0, nframes1 = len(video0["video"]), len(video1["video"])
        time0, time1 = nframes0 / fps0, nframes1 / fps1
        frame_time0_ = np.linspace(0, time0, nframes0)
        frame_time0 = ",".join([f"{i:.2f}s" for i in frame_time0_])
        frame_time1_ = (
            np.linspace(0, time1, nframes1) + time0
        )  # add time from first vid
        frame_time1 = ",".join([f"{i:.2f}s" for i in frame_time1_])

        # combine the 2 vids into a single vid
        video0, video1 = match_video_dimensions(video0["video"], video1["video"])
        prompt_videos = [torch.from_numpy(np.vstack([video0, video1]))]

        time_instruciton = f"The video is two different videos concatenated together.\n"
        time_instruciton += f"The first {nframes0} frames are video a, lasts for {time0:.2f} seconds, and these frames are located at {frame_time0}."
        time_instruciton += f"The second {nframes1} frames are video b, lasts for {time1:.2f} seconds, and these frames are located at {frame_time1}."
        text = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{prompt_text}"

        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        prompt_text = prompt_question

    else:
        raise ValueError(
            f"Config for lmm.video_representation [{video_representation}] not recognised"
        )

    return prompt_text, prompt_videos


def run_lmm(
    batch_prompts_text: list[str],
    batch_prompts_video: list[list[np.ndarray]],
    args_lmm: BaseContainer,
    eval_mode: int,
    n_differences: list[int],
    debug=None,
    overwrite_cache: bool = False,
    verbose: bool = True,
):
    """
    Assumes that the `batch_prompts_video` was formatted in an appropriate way
    for each api. For example, openai takes videos as sequeneces
    of images, so the `batch_prompts_video` is actually a list of images, and the
    text prompts in `batch_prompts_text` should explain that.
    """
    assert n_differences is not None
    json_mode = True

    if "gpt" in args_lmm.model:
        from apis import openai_api

        assert args_lmm.video_representation in ("frames", "first_frame")
        seeds = [args_lmm.seed] * len(batch_prompts_text)
        if verbose:
            logging.info(
                f"Running model {args_lmm.model} on {len(batch_prompts_text)} prompts"
            )
        res = openai_api.call_gpt_batch(
            batch_prompts_text,
            batch_prompts_video,
            seeds=seeds,
            model=args_lmm.model,
            debug=debug,
            overwrite_cache=overwrite_cache,
            json_mode=json_mode,
        )
        cost = sum([b[1] for b in res])
        logging.info(f"Cost for lmm differences generation: ${cost:.4f}")
        predictions = [b[0] for b in res]

    elif "claude" in args_lmm.model:
        from apis import openai_api

        assert args_lmm.video_representation in ("frames", "first_frame")
        seeds = [args_lmm.seed] * len(batch_prompts_text)
        if verbose:
            logging.info(
                f"Running model {args_lmm.model} on {len(batch_prompts_text)} prompts"
            )
        json_mode = False  # bc of the thing
        res = openai_api.call_gpt_batch(
            batch_prompts_text,
            batch_prompts_video,
            seeds=seeds,
            model=args_lmm.model,
            debug=debug,
            json_mode=json_mode,
        )
        cost = sum([b[1] for b in res])
        logging.info(f"Cost for lmm differences generation: ${cost:.4f}")
        predictions = [b[0] for b in res]

        if eval_mode != 1:
            predictions = _reformat_malformed_json_prediction(predictions)
        else:
            predictions = [r for r in res[0]]

    elif "gemini" in args_lmm.model:

        if args_lmm.video_representation != "video":
            raise ValueError(
                "Gemini requires 'video' representation which affects the text prompt"
            )

        seeds = [args_lmm.seed] * len(batch_prompts_text)
        res = gemini_api.call_gemini_batch(
            batch_prompts_text,
            batch_prompts_video,
            seeds=seeds,
            model=args_lmm.model,
            debug=debug,
            fps=args_lmm.fps_gemini,
        )
        if eval_mode != 1:
            predictions = _reformat_malformed_json_prediction([r for r in res[0]])
        else:
            predictions = [r for r in res[0]]

    elif "Qwen2-VL-7B-Instruct" in args_lmm.model:
        from apis import qwen_api

        if args_lmm.video_representation != "video":
            raise ValueError(
                "Qwen requires 'video' representation which affects the text prompt"
            )

        seeds = [args_lmm.seed] * len(batch_prompts_text)
        if verbose:
            logging.info(
                f"Running model {args_lmm.model} on {len(batch_prompts_text)} prompts"
            )
        msgs, responses = qwen_api.call_qwen_batch(
            batch_prompts_text,
            batch_prompts_video,
            seeds=seeds,
            model=args_lmm.model,
            debug=debug,
            json_mode=json_mode,
        )
        predictions = _reformat_malformed_json_prediction(msgs)

    elif "lmms-lab/LLaVA-Video-7B-Qwen2" in args_lmm.model:
        from apis import llavavideo_api

        assert args_lmm.video_representation == "llavavideo"

        seeds = [args_lmm.seed] * len(batch_prompts_text)
        if verbose:
            logging.info(
                f"Running model {args_lmm.model} on {len(batch_prompts_text)} prompts"
            )
        msgs, responses = llavavideo_api.call_llavavideo_batch(
            batch_prompts_text,
            batch_prompts_video,
            seeds=seeds,
            model=args_lmm.model,
            debug=debug,
        )
        try:
            predictions = [json.loads(m) for m in msgs]
        except json.JSONDecodeError as e:
            predictions = _reformat_malformed_json_prediction(msgs, cache=True)

    else:
        raise ValueError(
            f"Could not find an implementation for requested model [{args_lmm.model}]"
        )

    if eval_mode == "closed":
        predictions_final = predictions

    elif eval_mode == "open":
        predictions_final = _truncate_too_many_preds(
            predictions, n_differences, do_warning=True
        )

    else:
        raise ValueError()

    return predictions_final


def _remove_trailing_commas_json(json_string):
    """Some lmm outputs add a trailing string sometimes"""
    # Remove trailing commas from objects
    json_string = re.sub(r",(\s*})", r"\1", json_string)

    # Remove trailing comma from the last object in the main object
    json_string = re.sub(r",(\s*})$", r"\1", json_string)

    return json_string


def _reformat_malformed_json_prediction(malformed_outputs, skip=False, cache=True):
    cache_reformat = lmdb.open("cache/cache_reformat", map_size=int(1e12))

    # run the skip branch if we have high confidence the json will be correct
    if skip:
        predictions = []
        for pred in malformed_outputs:
            pred_dict = json.loads(_remove_trailing_commas_json(pred))
            predictions.append(pred_dict)
        return predictions

    prompts = [
        lp.prompt_reformat_malformed_json.replace("{llm_output}", g)
        for g in malformed_outputs
    ]
    seeds = [0] * len(prompts)
    res = openai_api.call_gpt_batch(
        prompts,
        seeds=seeds,
        model="gpt-4o-mini",
        max_tokens=4000,
        cache_dir=cache_reformat,
        cache=cache,
    )
    predictions = [r[0] for r in res]

    return predictions


def _truncate_too_many_preds(predictions, n_differences: list[int], do_warning: bool):
    """
    Just naiveley take the first `n_differences` values
    """
    for i, pred in enumerate(predictions):
        if len(pred) > n_differences[i]:

            if do_warning:
                logging.warning(
                    f"Max {n_differences[i]} differences allowed, but "
                    f"prediction {i} has {len(pred)}. Doing naive truncation."
                )

            predictions[i] = dict(list(pred.items())[: n_differences[i]])

    # double check that it worked
    assert all(
        [len(pred) <= n_diff for pred, n_diff in zip(predictions, n_differences)]
    )

    return predictions


def match_video_dimensions(video0: np.ndarray, video1: np.ndarray):
    T0, H0, W0, C0 = video0.shape
    T1, H1, W1, C1 = video1.shape
    if H0 != H1 or W0 != W1:
        transform = T.Compose([T.Resize((H0, W0))])
        video1 = (
            torch.stack(
                [
                    transform(torch.from_numpy(frame).permute(2, 0, 1))
                    for frame in video1
                ]
            )
            .permute(0, 2, 3, 1)
            .numpy()
        )

    return video0, video1

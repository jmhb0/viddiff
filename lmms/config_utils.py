from typing import Dict
from pathlib import Path
from omegaconf import OmegaConf
import os
import json


def load_config(
    f_config: str,
    f_cfg_base="lmms/configs/config.yaml",
    name=None,
    seed=None,
    eval_mode=None,
    split=None,
    model=None,
    video_representation=None,
    test_flip=False,
    subset_mode=None,
) -> Dict:
    """
    For each option 'None' means to use the value from the config files.
    """
    base_cfg = OmegaConf.load(f_cfg_base)
    if f_config != "":
        cfg = OmegaConf.load(f_config)
        final_cfg = OmegaConf.merge(base_cfg, cfg)
    else:
        final_cfg = base_cfg
        print(f"Using base config at {f_cfg_base}")

    # modify based on the command line options
    if name is not None:
        final_cfg.logging.name = name
    if eval_mode is not None:
        final_cfg.eval_mode = eval_mode
    if split is not None:
        final_cfg.data.split = split
    if model is not None:
        final_cfg.lmm.model = model
    if subset_mode is not None:
        final_cfg.data.subset_mode = subset_mode

    # special video representations for gemini, qwen and llavavideo
    if "gemini" in model.lower() or "qwen" in model.lower():
        final_cfg.lmm.video_representation = "video"
    if "llava-video" in model.lower():
        final_cfg.lmm.video_representation = "llavavideo"

    # create args object, resolving variable references
    args = OmegaConf.to_container(final_cfg, resolve=True)
    args = OmegaConf.create(args)
    args.config = f_config

    # create results dir. If it exists, throw error if logging.overwrite_okay=False
    if not args.logging.overwrite_ok and os.path.exists(args.logging.results_dir):
        raise ValueError(
            f"results_dir [{args.logging.results_dir}] already exists and "
            "config.logging.overwrite_ok is False."
        )
    os.makedirs(args.logging.results_dir, exist_ok=True)

    args.data.subset_mode = str(args.data.subset_mode)

    if "test_samevideo" not in args.keys():
        args.test_samevideo = 0
    if "test_flipvids" not in args.keys():
        args.test_flipvids = 0
    if args.test_flipvids:
        print("*" * 80)
        print("Flag set [args.test_flipvids] flips the vids")
        print(" but does NOT flip the A/B prediciton. Need to handle in postprocessing")
        print("*" * 80)

    # save config
    with open(os.path.join(args.logging.results_dir, "args.json"), "w") as f:
        json.dump(OmegaConf.to_container(args), f, indent=4)

    return args

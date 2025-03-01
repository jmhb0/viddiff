from typing import Dict
from pathlib import Path
from omegaconf import OmegaConf
import os
import json

def load_config(f_config: str,
                f_cfg_base="viddiff_method/configs/config.yaml",
                name=None,
                seed=None,
                split=None,
                eval_mode=None,
                test_flip=False,
                subset_mode=None) -> Dict:
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
    if seed is not None:
        final_cfg.seed = seed
    if split is not None:
        final_cfg.data.split = split
    if eval_mode is not None:
        final_cfg.eval_mode = eval_mode
        final_cfg.proposer.eval_mode = eval_mode
        final_cfg.retriever.eval_mode = eval_mode
        final_cfg.frame_differencer.eval_mode = eval_mode
    if subset_mode is not None:
        final_cfg.data.subset_mode = subset_mode

    # create args object, resolving variable references
    args = OmegaConf.to_container(final_cfg, resolve=True)
    args = OmegaConf.create(args)
    args.config = f_config

    if not final_cfg.logging.overwrite_ok and os.path.exists(
            final_cfg.logging.results_dir):
        raise ValueError(
            f"results_dir [{args.logging.results_dir}] already exists and "\
            "config.logging.overwrite_ok is False."
        )

    # create results dir. If it exists, throw error if logging.overwrite_okay=False
    if not args.logging.overwrite_ok and os.path.exists(
            args.logging.results_dir):
        raise ValueError(
            f"results_dir [{args.logging.results_dir}] already exists and "\
            "config.logging.overwrite_ok is False."
        )
    os.makedirs(args.logging.results_dir, exist_ok=True)

    args.data.subset_mode = str(args.data.subset_mode)
    # save config
    with open(os.path.join(args.logging.results_dir, "args.json"), 'w') as f:
        json.dump(OmegaConf.to_container(args), f, indent=4)

    return args
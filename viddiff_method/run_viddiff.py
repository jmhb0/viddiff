import ipdb
import click
from typing import Dict, List, Tuple
import logging
from line_profiler import LineProfiler

import sys

sys.path.insert(0, "")

# from viddiff_method import load_viddiff_dataset as lvd
from data import load_viddiff_dataset as lvd
from viddiff_method import config_utils
from viddiff_method.stage1_proposer import Proposer
from viddiff_method.stage2_retriever import Retriever
from viddiff_method.stage3_differencer import Differencer
from eval_viddiff import eval_viddiff

logging.getLogger("openai").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(filename)s:%(levelname)s:%(message)s")
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("_client").setLevel(logging.ERROR)


# yapf: disable
@click.command()
@click.option("--config", "-c", default="configs/base.yaml", help="config file")
@click.option("--name", "-n", default=None, help="experiment name which determines the filename. Default value uses the config file value")
@click.option("--seed", "-s", default=None, help="Random seed. Default value uses the config file value")
@click.option("--split", "-s", default=None, help="Data split. Default value uses the config file value")
@click.option("--eval_mode", "-e", default="closed", type=click.Choice(['closed','open']), help="Eval mode. Default value uses the config file value")
@click.option("--test_new", "-t", is_flag=True, default=False, help="Test new actions. Default value uses the config file value")
# @click.option("--test_flip", "-f", default=None, help="Flip the order of videos to test sensitivity to order. Default value uses the config file value")
@click.option("--subset_mode", "-s", default=None, help="Data subset mode (see configs/base.yaml). Default value uses the config file value")
# yapf enable
def main(config, name, seed, split, eval_mode, test_new, subset_mode):

    logging.info("Loading config")
    args = config_utils.load_config(config, name=name, seed=seed, split=split, eval_mode=eval_mode, subset_mode=subset_mode)

    logging.info(f"Loading dataset {args.data.split}")
    dataset = lvd.load_viddiff_dataset([args.data.split],
                                       args.data.subset_mode, 
                                       test_new=test_new)
    print("\nLength of dataset: ", len(dataset), "\n")

    videos = lvd.load_all_videos(dataset, do_tqdm=True, overwrite_cache=False)
    videos = lvd.downsample_videos(dataset, videos, args.fps_inference)

    n_differences = dataset['n_differences_open_prediction']

    logging.info(f"Running LLM proposer")
    proposer = Proposer(args.proposer, args.logging, dataset, n_differences)
    proposals = proposer.generate_proposals()

    logging.info(f"Running frame retrieval")
    retriever = Retriever(args.retriever, args.logging, dataset, videos, proposals)
    retrieved_frames = retriever.retrieve_frames()

    logging.info(f"Running VLM frame differencing")
    frame_diferencer = Differencer(args.frame_differencer, args.logging,
                                        dataset, videos, proposals,
                                        retrieved_frames)
    predictions = frame_diferencer.caption_differences()

    logging.info(f"Doing eval")
    results = eval_viddiff(dataset,predictions_unmatched=predictions,
                                        eval_mode=args.eval_mode,
                                        seed=args.seed,
                                        n_differences=n_differences,
                                        results_dir=args.logging.results_dir,
                                        diffs_already_matched=True,
                                        )
    print(results)

if __name__ == "__main__":
    main()

"""
Take an activity description and use an LLM to propose differences and action stages
"""

import ipdb
import numpy as np
import tqdm
import sys
import json
import logging
import copy
from pathlib import Path
from collections import OrderedDict
import itertools
from fuzzywuzzy import process as fw_process
from datasets import Dataset
from omegaconf.basecontainer import BaseContainer

from viddiff_method import prompts
from apis import openai_api
from proposer_types import Difference, Stage, Proposal, CustomJsonEncoder
import eval_viddiff

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Proposer:
    """
    Create self.proposals, which is a dict from the 'sample_key' of each sample
    to a Proposer object, which has:
    - proposed differences
    - sub-action stages
    - each difference is associated to one or more stages
    """

    def __init__(
        self,
        args: BaseContainer,
        args_logging: BaseContainer,
        dataset: Dataset,
        n_differences: list[int],
    ):
        """ """
        # save configs
        self.args = args
        self.verbose = self.args.verbose

        # save dataset, and make sure samples are ordered
        self.dataset = dataset
        self.n_differences = n_differences

        # logging subdirectory
        self.results_subdir = Path(args_logging.results_dir) / "stage1_proposer"
        self.results_subdir.mkdir(exist_ok=True, parents=True)

    def generate_proposals(self):

        # construct candidate differences. If eval_mode>0, read from gt
        if self.args.eval_mode == "open":
            self.query_1_differences()
        elif self.args.eval_mode == "closed":
            self.query_1_differences_gt()
        else:
            raise ValueError(f"Invalid eval_mode: {self.args.eval_mode}")

        # get text retrieval information for action stages
        self.query_2_subactions()  # actually has query 2 and query 3

        # link the differences to stages
        self.query_4_linking()

        if self.args.do_eval and self.args.eval_mode == "open":
            self.match_differences()

        return self.proposals

    def query_1_differences_gt(self):
        """For eval_mode 'closed', the gt differences are given"""
        self.responses_1_differences = {}
        for row in self.dataset:
            # get the differences that have a prediction for
            keys_gt = {k for k, v in row["differences_gt"].items() if v is not None}
            diffs = {
                k: v for k, v in row["differences_annotated"].items() if k in keys_gt
            }
            self.responses_1_differences[row["sample_key"]] = diffs

    def query_1_differences(self):
        """
        If open eval mode, make one 'differece' proposal per sample.
        We could have done one call per unique action, but we do one per sample
        instead.

        That's because, if you did one-per action, and you failed to predict a gt
        difference, then ALL the samples in that action automatically have an error.

        This way, the variance in accuracy between runs is reduced.

        Each sample uses it's `sample_hash` attribute as part of its seed passed
        to the LLM, added to self.args.seed
        """
        # get and verify template
        template_differences = prompts.lookup_prompts_proposer_1_differences[
            self.args.prompt_key_1_differences
        ]
        # if self.args.n_differences:
        #     assert "{n_differences}" in template_differences

        # one query per sample
        batch_texts = []
        batch_seeds = []
        sample_keys = []
        for sample, n_diff in zip(self.dataset, self.n_differences):
            seed = self.args.seed + sample["sample_hash"]
            prompt = template_differences.replace(
                "{action}", sample["action_description"]
            )
            prompt = prompt.replace("{n_differences}", str(n_diff))

            batch_texts.append(prompt)
            batch_seeds.append(seed)
            sample_keys.append(sample["sample_key"])

        # call gpt for differences
        logging.info("GPT call: generating differences")
        llm_batch = openai_api.call_gpt_batch(
            batch_texts, model=self.args.model, seeds=batch_seeds
        )

        cost = sum([b[1] for b in llm_batch])
        logging.info(f"Cost for difference generation: ${cost:.4f}")
        responses = [b[0] for b in llm_batch]

        # enforce max n_differences
        for res, n_diff in zip(responses, self.n_differences):
            if len(res) > n_diff:
                # logging.warning(f"\nA proposal had [{len(res)}] differences " \
                # f"but max allowed is {n_diff}")
                res = dict(list(res.items())[:n_diff])

        # log results to object and to file
        self.responses_1_differences = dict(zip(sample_keys, responses))
        self.results_subdir_differences = self.results_subdir / "1_differences"
        self.results_subdir_differences.mkdir(exist_ok=True, parents=True)
        for sample, res in zip(self.dataset, responses):
            f_save = (
                self.results_subdir_differences
                / f"sample_{sample['sample_key']}_action_{sample['action']}.json"
            )
            with open(f_save, "w") as f:
                json.dump(res, f, indent=4)

    def query_2_subactions(self):
        """
        Query one action at a time.
        """
        template_subactions = prompts.lookup_prompts_proposer_2_subactions[
            self.args.prompt_key_2_subactions
        ]
        # action_descriptions = list(
        #     self.dataset['lookup_idx_to_actions'].values())

        # get prompts
        batch_texts = []
        batch_seeds = []
        sample_keys = []
        action_descriptions = []
        for sample in self.dataset:
            seed = self.args.seed + sample["sample_hash"]
            prompt_subactions = template_subactions.replace(
                "{action}", sample["action_description"]
            )
            prompt_subactions = prompt_subactions.replace(
                "{n_retrieval_keys}", str(self.args.n_retrieval_keys)
            )

            action_descriptions.append(sample["action_description"])
            sample_keys.append(sample["sample_key"])
            batch_texts.append(prompt_subactions)
            batch_seeds.append(seed)

        # run
        logging.info("GPT call: generating subactions")
        llm_batch = openai_api.call_gpt_batch(
            batch_texts, seeds=batch_seeds, model=self.args.model
        )
        cost = sum([b[1] for b in llm_batch])
        responses = [b[0] for b in llm_batch]
        logging.info(f"Cost for stages generation: ${cost:.4f}")

        # log
        self.results_subdir_subactions = self.results_subdir / "2_subactions"
        self.results_subdir_subactions.mkdir(exist_ok=True, parents=True)

        self.responses_2_stages_ = dict(zip(sample_keys, responses))
        for sample, res in zip(self.dataset, responses):
            f_out = (
                self.results_subdir_subactions
                / f"sample_{sample['sample_key']}_action_{sample['action']}_subactions_first.json"
            )
            with open(f_out, "w") as f:
                json.dump(res, f, indent=4)

        # optionally filter bad retrieval keys
        if self.args.filter_retrieval_keys:
            # prompts
            template_subactions_refine = (
                prompts.lookup_prompts_proposer_2_subactions_refiner[
                    self.args.prompt_key_3_subaction_filtering
                ]
            )
            batch_texts = []
            for sample_key, sample in zip(sample_keys, self.dataset):

                response_old = self.responses_2_stages_[sample_key]
                prompt_refine = template_subactions_refine.replace(
                    "{action}", sample["action_description"]
                )
                prompt_refine = prompt_refine.replace(
                    "{stages}", json.dumps(response_old, indent=4)
                )
                batch_texts.append(prompt_refine)

            logging.info("GPT call: filtering retrieval keys")
            llm_batch = openai_api.call_gpt_batch(
                batch_texts, model=self.args.model, seed=self.args.seed
            )
            cost = sum([b[1] for b in llm_batch])
            logging.info(f"Cost for retrieval key filtering: ${cost:.4f}")
            responses = [b[0] for b in llm_batch]

            # log
            self.responses_2_stages = dict(zip(sample_keys, responses))
            for sample, res in zip(self.dataset, responses):
                f_out = (
                    self.results_subdir_subactions
                    / f"sample_{sample['sample_key']}_action_{sample['action']}_subactions_refined.json"
                )
                with open(f_out, "w") as f:
                    json.dump(res, f, indent=4)
        else:
            # if not doing the filtering
            self.responses_2_stages = self.responses_2_actionkey_to_stages_

    def query_4_linking(self):
        """ """

        template_linking = prompts.lookup_prompts_proposer_3_linking[
            self.args.prompt_key_4_linking
        ]

        batch_texts = []
        for sample in self.dataset:
            sample_key = sample["sample_key"]

            stages = self.responses_2_stages[sample_key]
            differences = self.responses_1_differences[sample_key]

            prompt_linking = template_linking.replace(
                "{action}", sample["action_description"]
            )
            prompt_linking = prompt_linking.replace(
                "{stages}", json.dumps(stages, indent=4)
            )
            prompt_linking = prompt_linking.replace(
                "{differences}", json.dumps(differences, indent=4)
            )

            batch_texts.append(prompt_linking)

        # call llm
        logging.info("GPT call: linking")
        llm_batch = openai_api.call_gpt_batch(
            batch_texts, model=self.args.model, seed=self.args.seed
        )
        cost = sum([b[1] for b in llm_batch])
        logging.info(f"Cost for linking generation: ${cost:.4f}")
        responses = [b[0] for b in llm_batch]

        sample_keys = self.dataset["sample_key"]
        self.responses_3_linking = dict(zip(sample_keys, responses))

        # log vlm response
        self.results_subdir_linking = self.results_subdir / "4_linking"
        self.results_subdir_linking.mkdir(exist_ok=True, parents=True)
        for sample_key, res in self.responses_3_linking.items():
            f_out = self.results_subdir_linking / f"sample_{sample_key}_linking.json"
            with open(f_out, "w") as f:
                json.dump(res, f, indent=4)

        # construct the final Proposer object using the `link` llm response
        self.proposals = {}
        for i, sample in enumerate(self.dataset):
            sample_key = sample["sample_key"]
            differences = self.responses_1_differences[sample_key]
            stages = self.responses_2_stages[sample_key]
            links = self.responses_3_linking[sample_key]

            ## we do some validation some basic validation checks.
            # Reminder: the `link` keys are stage names, and the values are lists of differences
            stage_names = [d["name"] for d in stages["stages"]]
            difference_names = set([d["name"] for d in differences.values()])

            # Issue 1: a stage name in the link keys is hallucinated
            hallucinated_stages_in_links = set(links.keys()) - set(stage_names)
            if len(hallucinated_stages_in_links) > 0:
                # logging.warning(
                #     f"\nllm response has bad stage keys: {hallucinated_stages_in_links}\nReal stage links real: {stage_names}"
                # )
                for h_stage in hallucinated_stages_in_links:

                    # if it's very close in edit distance to another stage, then just add it to that stage
                    stage_match, score = fw_process.extractOne(h_stage, stage_names)
                    if score > 80:
                        links[stage_match] = links[h_stage]
                    # otherwise just delete it. It's okay if there are differences missing from `links` ... there's a check for that later fix any spelling errors
                    else:
                        pass
                    del links[h_stage]

            # Issue 2: a stage is missing from the links list. Just add an empty list
            missing_stages = set(stage_names) - set(links.keys())
            if len(missing_stages) > 0:
                for s in missing_stages:
                    links[s] = []

            # iterate over stages and put the corresponding differences in the stage dict

            for stage in stages["stages"]:
                ## issue 2: one of the linked differences is not in the original proposed differences. Rename it to the nearest string match
                hallucinated_link_diffs = set(links[stage["name"]]) - set(
                    difference_names
                )
                if len(hallucinated_link_diffs) > 0:
                    for h_diff in hallucinated_link_diffs:
                        match, _ = fw_process.extractOne(h_diff, difference_names)
                        # differences_linked = set(differences_linked) - {h_diff} | {match}
                        links[stage["name"]] = list(
                            set(links[stage["name"]]) - {h_diff} | {match}
                        )
                # double check that any corrections do work
                hallucinated_link_diffs = set(links[stage["name"]]) - set(
                    difference_names
                )
                assert len(hallucinated_link_diffs) == 0
                stage["differences"] = links[stage["name"]]

            # issue 3: there were differences from the proposal that were missed in linking. Just assign it arbitrarily to the middle stage
            linked_differences_names = set(sum(links.values(), []))
            missing_diffs = difference_names - linked_differences_names
            if len(missing_diffs) > 0:
                # logging.warning(f"\nMissing some differences in the linking. \nDifferences were:\n{difference_names}\nLinked differences were:\n{linked_differences_names}\nMissing:\n{missing_diffs}"\
                #     f"\nAssigning to the middle stage in sample {sample_key} action {sample['action_name']}")
                n_stages = len(stages["stages"])
                for diff in missing_diffs:
                    stages["stages"][n_stages // 2]["differences"].append(diff)

            # construct the differnece, stages, and proposal objects
            differences = {k: Difference(**var) for k, var in differences.items()}
            stages = [Stage(**stage) for stage in stages["stages"]]
            proposal = Proposal(
                action_key=sample["action"],
                action_description=sample["action_description"],
                stages=stages,
                differences=differences,
            )
            proposal.postprocess()
            self.proposals[sample["sample_key"]] = proposal

    def _validate_linking_operations(self, differences, links):
        # ipdb.set_trace()
        pass

    def match_differences(self):
        """
        In eval mode, find the correspondence between proposed and gt variations.
        If self.args.drop_unmatched_diffs=True (and it is by default) then drop
        any variations that had no matches. This avoids computational expense
        of stuff that won't be evaluated anyway.

        Warning: if `self.args.drop_unmatched_diffs`, then unmatched difference
        keys get removed. That's fine in this framework because from this point
        on, no candidate variation affects any other. But if that changes, then
        it would no longer be correct to delete it. E.g. maybe in the last stage
        of the system, you want to make a query about multiple variations at
        once.
        """

        # create a predictions dict that works with the eval_viddiff functions
        proposals_for_matching = []
        for sample in self.dataset:
            differences = self.proposals[sample["sample_key"]].differences
            diff_set = {
                k: {"description": v["description"]} for k, v in differences.items()
            }
            proposals_for_matching.append(diff_set)

        # do the matching
        matching, predictions_excess = eval_viddiff.do_matching(
            self.dataset, proposals_for_matching, self.args.seed
        )

        # remove the differences that wre not in the matching results. `matching` has an element for
        # every gt difference. If there's a matching `pred` then `pred_key` will be a key and not "None"
        def _clean_dict(d):
            keys_keep = ("pred_description", "gt_description", "pred_key")
            return {
                k: {key: value for key, value in v.items() if key in keys_keep}
                for k, v in d.items()
            }

        matching = [_clean_dict(item) for item in matching]

        # filter out things that don't get matched
        self.results_subdir_matching = self.results_subdir / "5_matching"
        self.results_subdir_matching.mkdir(exist_ok=True, parents=True)
        with open(self.results_subdir_matching / "matching.json", "w") as fp:
            json.dump(matching, fp, indent=4)

        # log the differences that wre discarded (they are excess)
        with open(
            self.results_subdir_matching / "non_matched_differences.json", "w"
        ) as fp:
            predictions_excess_dict = {
                row["sample_key"]: pred
                for (row, pred) in zip(self.dataset, predictions_excess)
            }
            json.dump(predictions_excess_dict, fp, indent=4)

        # remove differences from proposal.differences that were not matched. Remap the difference keys to the gt key.
        num_gt = 0
        num_preds_after_matching = 0
        for i, sample in enumerate(self.dataset):
            proposal = self.proposals[sample["sample_key"]]
            num_gt += len(matching[i])
            proposal.remap_indexes(matching[i])
            num_preds_after_matching += len(proposal.differences)

        acc_recalled = num_preds_after_matching / num_gt
        logging.info(
            f"\nRecovered {acc_recalled:.4f} of the differences in the matching\n"
        )

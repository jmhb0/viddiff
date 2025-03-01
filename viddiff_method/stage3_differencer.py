"""
"""
import ipdb
import numpy as np
import textwrap
import json
from PIL import Image
import tqdm
from pathlib import Path
from typing import List
from collections import Counter
import copy
from collections import defaultdict
import time
import logging
import re

from apis.openai_api import call_gpt, call_gpt_batch
from viddiff_method import prompts as prompt_templates
from viddiff_method import utils


class Differencer():
    """
    Creates `self.sample_predictions` so that  
        self.sample_predictions[k0][k1]

    Is sample k0, difference k1, and is a dict with required fields:
        'pred' (the prediction)
    And any other optional fields, for example: 
        'confidence': the vlm's estimated confidence 
        'query_string': string description used by the vlm 

    The difference key, k1, is defined in the output of the proposer stage, which 
    is stored in proposals. Since the differences are the same for
    each action, it can be looked up in proposals[sample_key]. 
    """

    def __init__(self, args, args_logging, dataset, videos, proposals,
                 retrieved_frames):
        self.args = args
        self.args_logging = args_logging
        self.dataset = dataset
        self.videos0 = videos[0]
        self.videos1 = videos[1]
        self.proposals = proposals
        self.retrieved_frames = retrieved_frames

        # logging subdirectory
        self.results_subdir = Path(
            args_logging.results_dir) / "stage3_differencer"
        self.results_subdir.mkdir(exist_ok=True, parents=True)

        self.sample_key_to_differences_gt = {
            item['sample_key']: item['differences_gt']
            for item in self.dataset
        }

        # self.prompt_template_1frame = lookup_prompts_differencing_1_frame[
        #     self.args.prompt_key]
        # self.prompt_template_multiframe = lookup_prompts_differencing_multiframe[
        #     self.args.prompt_key_multiframe]
        # pass

    def caption_differences(self):

        self.prepare_prompts()
        self.call_vlm()
        self.make_final_predictions()

        return self.predictions_for_eval

    def prepare_prompts(self):
        """ 
        Prepare prompts: the frames and the text. 
        Put it in self.prompts that has the same keys as self.dataset['samples'].

        Each self.prompts[key] has keys 'prompt_frames' and `prompt_text`, which
        are lists that are the length of the number of action stages. 
        """
        self.prompts = {}
        self.metas = {}  # for logging

        for sample, video0, video1 in zip(self.dataset, self.videos0,
                                          self.videos1):
            sample_key = sample['sample_key']
            self.prompts[sample_key] = {}
            self.metas[sample_key] = {}

            # difference info
            proposal = self.proposals[sample_key]
            fps0 = video0['fps']
            fps1 = video1['fps']
            assert fps0 == fps1

            # frame retrieval info. Group together queries that have the same frames
            retrieve_frames_0 = self.retrieved_frames[sample_key][0]
            retrieve_frames_1 = self.retrieved_frames[sample_key][1]

            # prepare the prompts for this sample
            prompts_text = []
            prompts_frames = []
            prompts_difference_idxs = []
            metas = []

            assert retrieve_frames_0.keys() == retrieve_frames_1.keys()

            for difference_idx in retrieve_frames_0.keys():

                prompts_difference_idxs.append(difference_idx)

                # get the retrieved frames
                frame_idxs_video0 = retrieve_frames_0[difference_idx]
                frame_idxs_video1 = retrieve_frames_1[difference_idx]

                frames_0 = list(video0['video'][frame_idxs_video0])
                frames_1 = list(video1['video'][frame_idxs_video1])
                nframes = len(frames_0)

                prompts_frames.append(frames_0 + frames_1)

                ## make the text prompts
                # single frame prompt
                if len(frames_0) == 1:
                    prompt = prompt_templates.lookup_prompts_differencing_1_frame[
                        self.args.prompt_key]
                    
                    num_frames = 1
                    time_diff = None
                    prompt = prompt.replace(
                        "{query_string}",
                        proposal.differences[difference_idx]['description'])

                # multiframe prompt
                else:
                    prompt = prompt_templates.lookup_prompts_differencing_multiframe[
                            self.args.prompt_key_multiframe]
                    
                    num_frames = len(frames_0)
                    frame_sep = frame_idxs_video0[1] - frame_idxs_video0[0]
                    time_diff = frame_sep / fps0
                    prompt = prompt.replace("{num_frames}", str(num_frames))
                    prompt = prompt.replace("{time_diff}", f"{time_diff:.2f}")
                    prompt = prompt.replace(
                        "{query_string}",
                        proposal.differences[difference_idx]['description'])

                query_string = proposal.differences[difference_idx][
                    'query_string']
                prompt = prompt.replace("{action}",
                                        proposal.action_description)
                prompt = prompt.replace("{query_string}", query_string)

                prompts_text.append(prompt)
                # just for logging
                metas.append({
                    "num_frames": 1,
                    "time_diff": None,
                    "action": proposal.action_description,
                    "query_string": query_string
                })

            # save prompt text and frames for this data pair, save the prompts
            self.prompts[sample_key]['prompts_text'] = prompts_text
            self.prompts[sample_key]['prompts_frames'] = prompts_frames
            self.prompts[sample_key]['stages'] = proposal.stages
            self.prompts[sample_key]['action'] = proposal.action_key
            self.prompts[sample_key][
                'difference_idxs'] = prompts_difference_idxs
            self.metas[sample_key] = metas

    def call_vlm(self):
        """ 
        For each stage in the action transcript (that has an associated 'difference')
        call a vlm. 

        Save predictions to self.samples_preds_stages so that for sample video 
        pair with `key`, we have self.samples_preds_stages[key], which is a list
        of dicts. Each dict has the vlm response
        """
        self.samples_preds_vlm = {}

        # gpt kwargs
        system_prompt = prompt_templates.lookup_system_prompt_vision[
            self.args.system_prompt_key]
        kwargs_gpt = dict(
            seed=self.args.seed,
            model=self.args.model,
            system_prompt=system_prompt,
        )

        # put all the texts and frames into a single batch
        batch_texts = []
        batch_frames = []

        batch_idx = 0
        # iterate over samples
        for i, (key, prompts) in enumerate(self.prompts.items()):
            for j, (imgs, text, difference_idx) in enumerate(
                    zip(prompts['prompts_frames'], prompts['prompts_text'],
                        prompts['difference_idxs'])):
                batch_texts.append(text)
                batch_frames.append(imgs)
                batch_idx += 1
        # ipdb.set_trace()

        logging.info(f"Calling GPT batch mode: {len(batch_texts)} queries")

        start = time.time()
        json_mode = True 
        vlm_results_batch = call_gpt_batch(texts=batch_texts,
                                           imgs=batch_frames,
                                           json_mode=json_mode,
                                           **kwargs_gpt)
        duration = time.time() - start
        logging.info(f"time taken: {int(duration)}s")
        cost = sum([v[1] for v in vlm_results_batch])
        logging.info(f"vlm call cost: ${cost:.4f}")

        logging.info(f"logging vlm results")
        batch_idx = 0

        for i, (key, prompts) in enumerate(self.prompts.items()):
            preds = []
            for j, (imgs, text, difference_idx) in enumerate(
                    zip(prompts['prompts_frames'], prompts['prompts_text'],
                        prompts['difference_idxs'])):
                msg = vlm_results_batch[batch_idx]
                preds_stage = self.log_vlm_call(self.args.log_imgs, j,
                                                difference_idx, key, imgs,
                                                text, msg, prompts['action'])
                preds.append(preds_stage)
                batch_idx += 1
            self.samples_preds_vlm[key] = preds

    def log_vlm_call(self, log_imgs, i, difference_idx, key, imgs, text, msg,
                     action):
        """ 
        Preare the results dictionary.
        Logs the image repsonse to file only if `log_imgs=True`.
        """

        results_subdir_this_pair = self.results_subdir / f"key_{key}_action_{action}"
        results_subdir_this_pair.mkdir(exist_ok=True)
        assert type(
            difference_idx) is str, "expect one difference per vlm call"

        f_prompt = results_subdir_this_pair / f"idxs_{difference_idx}_prompt.txt"
        f_response = results_subdir_this_pair / f"idxs_{difference_idx}_response.json"
        f_imgs = results_subdir_this_pair / f"idxs_{difference_idx}_imgs.png"
        f_both = results_subdir_this_pair / f"idxs_{difference_idx}_response.png"

        # if we know what the gt answer is, then get it.

        if log_imgs:
            assert len(imgs) % 2 == 0, "expected odd number of images"
            imgs_pil = [Image.fromarray(img) for img in imgs]
            imgs_0 = imgs_pil[:len(imgs) // 2]
            imgs_1 = imgs_pil[len(imgs) // 2:]
            imgs_row_0 = utils.stack_images_seq(imgs_0, 'h')
            imgs_row_1 = utils.stack_images_seq(imgs_1, 'h')
            imgs_stack = utils.stack_images_seq([imgs_row_0, imgs_row_1], 'v')
            imgs_stack.save(f_imgs)

        with open(f_prompt, 'w') as f:
            f.write(text)

        # recover the gt value (used for logging). If eval is open, need a lookup to find the right gt difference
        if self.args.do_eval:
            differences_gt_all = self.sample_key_to_differences_gt[key]
            gt_label_this_difference = differences_gt_all[difference_idx]

        # log files
        # yapf: disable
        preds_log = {
            "difference_idx": difference_idx,
            "text_prompt": text,
            "response": msg,
            "gt_label_this_difference": gt_label_this_difference,
            # very hacky way to get this for logging ... mybad
            "difference_description" :self.proposals[key]['differences'][difference_idx]['description'],
        }
        # yapf: enable
        with open(f_response, 'w') as f:
            json.dump(preds_log, f, indent=4)

        preds = preds_log.copy()
        preds['imgs'] = imgs

        return preds

    def make_final_predictions(self):
        """ 
        Make the final predictions in the format consumable by the evaluator.

        We have self.samples_preds with the VLM call outputs.

        Required output object `preds` is a dict, preds
        """
        # this list is what we send to the evaluator
        self.predictions_for_eval = []

        # I forgot if these two objects are important as well ...
        self.sample_predictions = {}
        self.sample_predictions_verbose = {}

        # loop over sample
        for sample_key, preds_vlm in self.samples_preds_vlm.items():
            self.sample_predictions[sample_key] = {}

            # loop over the vlm cals within one sample
            for pred_vlm in preds_vlm:
                difference_key = pred_vlm['difference_idx']
                pred_vlm['pred'] = _force_pred(
                    pred_vlm['response'][0]['answer'])
                # pred_vlm['pred_detail'] = pred_vlm['response'][0][
                #     'answer_detailed']
                self.sample_predictions[sample_key][difference_key] = pred_vlm

            pred_eval = {
                pred_vlm['difference_idx']: {
                    'prediction': pred_vlm['pred'],
                    'description': pred_vlm['difference_description'],
                }
                for pred_vlm in preds_vlm
            }
            self.predictions_for_eval.append(pred_eval)


def _force_pred(r):
    """in case of llm failure which happens sometimes """
    if r in ('a', 'b', 'c'):
        return r
    else:
        logging.warning(f"Forced pred, input was {r}")
        return 'a'


def vote_on_predictions(lst, priority_order=['a', 'b', 'c', 'd']):
    """ 
    For a list of different predictions 'a|b|c|d' on a single target, return 
    the top-voted target with a given priority order
    """
    counter = Counter(lst)
    most_common = counter.most_common()
    max_count = most_common[0][1]

    # now get the most common while respecting the priority order
    max_elements = [elem for elem, count in most_common if count == max_count]
    priority_dict = {key: i for i, key in enumerate(priority_order)}
    max_elements_sorted = sorted(
        max_elements, key=lambda x: priority_dict.get(x, float('inf')))

    return max_elements_sorted[0]

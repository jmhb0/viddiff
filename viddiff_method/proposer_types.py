from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Dict, Any
import warnings
import ipdb
import logging
import json


class DictLikeClass:
    """Base class to be indexable like a dictionary: object['attribute']"""

    def __getitem__(self, item):
        return self.__dict__[item]


@dataclass
class Difference(DictLikeClass):
    """used by Proposer"""

    name: str
    query_string: str
    description: str
    num_frames: Literal["1", "2", "gt_2"]
    retrieval_stages: str = None  # optional used by in eval_mode 2
    label: Literal["a", "b", "c", None] = None
    idx: int = None

    class Config:
        extra = "forbid"  # more fields not allowed at construction

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self):
        return json.dumps(self.to_dict())


@dataclass
class Stage(DictLikeClass):
    """used by Proposer"""

    name: str
    description: str
    retrieval_strings: List[str]
    differences: List[str]

    class Config:
        extra = "forbid"  # more fields not allowed at construction


@dataclass
class Proposal(DictLikeClass):
    """
    Holds the response from the Lll of type LlmProposal and also has some
    extra validation checking
    """

    action_key: str
    action_description: str
    stages: List[Stage]
    differences: Dict[str, Difference]
    # in eval_mode==0, store the original proposal from before matching
    differences_proposed: Dict[str, Difference] = None
    lookup_difference_name_to_dict: Dict[str, Any] = field(default_factory=dict)

    def postprocess(self, input_differences=None):
        """
        This needs to be manually called after creating the data class.
        Creates some useful objects.

        Args:
            input_differences: for closed eval (eval_mode>0), the prompt variations
        """
        # make the dictionary from difference_name -> difference object. Also add idx property
        for diff_idx, difference in self.differences.items():
            difference.idx = diff_idx
            name = difference["name"]
            self.lookup_difference_name_to_dict[name] = difference

        # make a map from difference_name->stages where that difference is predicted to exist
        # make the same map difference_idx->stages
        self.lookup_diffname_to_stages = {
            k["name"]: [] for k in self.differences.values()
        }
        self.lookup_diffidx_to_stages = {k: [] for k in self.differences.keys()}

        # make dictionary from difference idx -> ALL the stages where it's relevant
        for stage in self.stages:
            for diffname in stage.differences:
                self.lookup_diffname_to_stages[diffname].append(stage["name"])
                diffidx = self.lookup_difference_name_to_dict[diffname]["idx"]
                self.lookup_diffidx_to_stages[diffidx].append(stage)

    def remap_indexes(self, matches):
        """
        The proposal differences will have keys in some order. After doing
        matching, remap the differences indexes to correspond with the gt indexes
        and delete anything that's unmatched.

        Also, `self.lookup_diffidx_to_stages`

        matches (dict) is the output of eval_viddiff.py, function `do_matching`
        """
        # map from difference key (current) to gt difference key (target)
        idx_mapping = {
            v["pred_key"]: k for k, v in matches.items() if v["pred_key"] != "None"
        }
        # update the proposal
        self.differences = {
            idx_mapping[k]: v for k, v in self.differences.items() if k in idx_mapping
        }

        # update the lookup dict from difference idxs -> stages
        self.lookup_diffidx_to_stages = {
            idx_mapping[k]: v
            for k, v in self.lookup_diffidx_to_stages.items()
            if k in idx_mapping
        }


# class for running json.dump on simple dataclasses
class CustomJsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, (Difference, Stage)):
            return obj.to_dict()
        return super().default(obj)

# Video action differencing benchmark (VidDiffBench) 
This is evaluation code for the VidDiffBench benchmark, [available on 🤗 HuggingFace here](https://huggingface.co/datasets/jmhb/VidDiffBench). It's from the ICLR 2025 paper [Video Action Differencing](https://jmhb0.github.io/viddiff). The below text introduces the task, and has evaluation code. The paper also proposed Viddiff method, which is in `viddiff_method` - read about at [this README](viddiff_method/README.md). 


# Task: Video Action Differencing
The Video Action Differencing task compares two videos of the same action. The goal is to identify differences in how the action is performed, where the differences are expressed in natural language.

![morecontent](https://raw.githubusercontent.com/jmhb0/jmhb0.github.io/main/images/pull%20fig-5.jpg)

In closed evaluation: 
- Input: two videos of the same action ($v_a, v_b$), action description string $s$, a list of candidate difference strings $\lbrace d_0, d_1, ...\rbrace$.
- Output: for each difference string $d_i$, predict $p_i\in\lbrace a,b\rbrace$, which is either 'a' if the statement applies more to video a, or 'b' if it applies more to video 'b'.

In open evaluation, the model must generate the difference strings:
- Input: two videos of the same action ($v_a, v_b$), action description string $s$, an integer $n_{\text{diff}}$.
- Output: a list of difference strings, $\lbrace d_0, d_1, ...\rbrace$, with at most $n_{\text{diff}}$ differences. For each difference string $d_i$, predict $p_i\in\lbrace a,b\rbrace$, which is either 'a' if the statement applies more to video a, or 'b' if it applies more to video 'b'.



## Get the dataset
Get `dataset` and `videos` from the Huggingface hub: [https://huggingface.co/datasets/jmhb/VidDiffBench](https://huggingface.co/datasets/jmhb/VidDiffBench)

## Evaluation
First: `pip install -r requirements.txt`

### Prediction format:
Collect `predictions` as a list of dicts, like this:
```
predictions = [
  {
    "difference_key": {
      "description": "...",
      "prediction": "a|b"
    }, 
    ... // other predictions for this sample
  },
  ... // other samples
]
```
- Prediction at `predictions[i]` is for the sample at `dataset[i]`. Since we have multiple differences to predict, the dictionary has multiple entries.
- The "difference_key" are the keys from `dataset[i]['differences_gt']`.
- The "prediction" is 'a' or 'b'. 
- The "description" is the text description of the difference (only used in open evaluation). 

For example:
```
predictions = [
  {
    "0": {
      "description": "the feet stance is wider",
      "prediction": "b"
    }, 
    "1": {
      "description": "the speed of hip rotation is faster",
      "prediction": "a"
    }, 
  },
  ... // other samples
]
```

For closed evaluation, you can skip the description field, and write it without the lowest-level dict:
```
predictions = [
  {
    "0": "b",
    "1": "a",
    },
  ... // other samples
]
```
### Running evaluation
For a `dataset` and `predictions` as above, run:
```
import eval_viddiff

eval_mode = "closed" # or "open"
results_dir="results/name_of_experiment" # Path or None
metrics = eval_viddiff.eval_viddiff(
	dataset,
	predictions,
	eval_mode=eval_mode,
	results_dir=results_dir,
	seed=0)
print(metrics)
```


### Open evaluation 
In open evaluation, the model must generate the difference strings, so we need to match the predicted "description" string to the ground truth description. This is handled in the `eval_viddiff.py` file, and uses an LLM evaluator. By default, it uses OpenAI API, and so you needs to set the `OPENAI_API_KEY` environment variable. 



## Running LMM predictions 

We tested VidDiffBench on some popular LMMs: GPT-4o, Claude, Gemini, QwenVL, and LLaVA-video:
```
python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_closed_easy --split easy --eval_mode closed --model gpt-4o-2024-08-06
```
For API calls, we cache responses, e.g. `cache/cache_openai`. Most caches are committed with this repo. The options above are the deafults. 

For --model option: 
- Openai API, e.g. we tested 'gpt-4o-2024-08-06', set OPENAI_API_KEY environment variable. 
- Openrouter API, e.g. we tested 'anthropic/claude-3-5-sonnet', set OPENROUTER_API_KEY environment variable. 
- Gemini API, e.g. we tested 'models/gemini-1.5-pro', set GEMINI_API_KEY environment variable. This one is really slow to run bc we didn't implement batching. 
- QwenVL and LLaVA-video we did not use an API, so you need to run it locally. Follow package installation instructions [from here for Qwen](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) and [from here for LLaVA-video](https://huggingface.co/lmms-lab/LLaVA-Video-72B-Qwen2). It's slow because no batching. 

The other options:
- `name` used to save the results to `lmms/results/<name>`
- `split` is the split to run on: `easy`, `medium`, `hard`, default is `easy`.
- `eval` is the evaluation mode: `closed`, `open`, default is `closed`.

The inference fps is controlled in the config file `lmms/configs/config.yaml`. We've implemented each model according to it's API. The text prompts are in `lmms/lmm_prompts.py`, which are the same for all models, except for a preamble that describes the video representation: e.g. GPT models are represented as frames, while Gemini is represented as video. We also implemented automatic caching of all LMM calls in `cache/`


## VidDiff method 
The Viddiff method is in `viddiff_method`. To run it, look at [this README](viddiff_method/README.md). 

## Acknowledgements 
Code for running CLIP servers taken from [VisDiff](https://github.com/Understanding-Visual-Datasets/VisDiff). The VidDiff method has a temporal action segmentation step that borrows from [Anna Kukleva's repo](https://github.com/Annusha/unsup_temp_embed) in `viddiff_method/utils_retriever.py`.


## Citation 
Please cite the paper: 
```

@inproceedings{burgessvideo,
  title={Video Action Differencing},
  author={Burgess, James and Wang, Xiaohan and Zhang, Yuhui and Rau, Anita and Lozano, Alejandro and Dunlap, Lisa and Darrell, Trevor and Yeung-Levy, Serena},
  booktitle={The Thirteenth International Conference on Learning Representations}
}```

If you used the benchmark, then also cite the papers where we sourced the videos: citations listed at the bottom of https://huggingface.co/datasets/jmhb/VidDiffBench

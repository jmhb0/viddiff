# Video action differencing benchmark (VidDiffBench) 
This is evaluation code for [the VidDiff benchmark](https://huggingface.co/datasets/jmhb/VidDiffBench), from the ICLR 2025 paper [Video Action Differencing](https://openreview.net/forum?id=3bcN6xlO6f). The below text introduces the task, and has evaluation code. The paper also proposed Viddiff method, which is in `viddiff_method` - read about at [this README](viddiff_method/README.md). 

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
TODO: pip install and so on 

### Prediction format:
Collect `predictions` as a list of dicts, like this 
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
TODO: explain the args a little more

We tested VidDiffBench on some popular LMMs: GPT-4o, Claude, Gemini,  QwenVL, and LLaVA-video:
```
python lmms/run_lmm.py --config lmms/configs/config.yaml --name gpt4o_closed_easy --split easy --eval closed --model gpt-4o-2024-08-06 --video_representation=frames
```
The options above are the deafults. 

For --model option: 
- Openai API, e.g. we tested 'gpt-4o-2024-08-06', set OPENAI_API_KEY environment variable. 
- Openrouter API, e.g. we tested 'anthropic/claude-3-5-sonnet', set OPENROUTER_API_KEY environment variable. 
- Gemini API, e.g. we tested 'models/gemini-1.5-pro', set GEMINI_API_KEY environment variable. This one is really slow to run bc we didn't implement batching. 
- QwenVL we did not use an API, so you need to run it locally. Click [here](apis/howto-local-models.md). Slow because no batching. 
- LLaVA video we did not find via API, so you need to run it locally. Click [here](apis/howto-local-models.md). Slow because no batching. 

The inference fps is controlled in the config file `lmms/configs/config.yaml`. We've implemented each model according to it's API. The text prompts are in `lmms/lmm_prompts.py`, which are the same for all models, except for a preamble that describes the video representation: e.g. GPT models are represented as frames, while Gemini is represented as video. We also implemented automatic caching of all LMM calls in `cache/`


## VidDiff method 
The Viddiff method is in `viddiff_method`. To run it, look at [this README](viddiff_method/README.md). 

## Citation 
Please cite the paper using, and also the papers where we sourced the videos (`\cite{burgessvideo, cai2022humman, parmar2022domain, grauman2024ego, gao2014jhu, }`).
```
@inproceedings{burgessvideo,
  title={Video Action Differencing},
  author={Burgess, James and Wang, Xiaohan and Zhang, Yuhui and Rau, Anita and Lozano, Alejandro and Dunlap, Lisa and Darrell, Trevor and Yeung-Levy, Serena},
  booktitle={The Thirteenth International Conference on Learning Representations}
}

@inproceedings{cai2022humman,
  title={{HuMMan}: Multi-modal 4d human dataset for versatile sensing and modeling},
  author={Cai, Zhongang and Ren, Daxuan and Zeng, Ailing and Lin, Zhengyu and Yu, Tao and Wang, Wenjia and Fan,
          Xiangyu and Gao, Yang and Yu, Yifan and Pan, Liang and Hong, Fangzhou and Zhang, Mingyuan and
          Loy, Chen Change and Yang, Lei and Liu, Ziwei},
  booktitle={17th European Conference on Computer Vision, Tel Aviv, Israel, October 23--27, 2022,
             Proceedings, Part VII},
  pages={557--577},
  year={2022},
  organization={Springer}
}
          
@inproceedings{parmar2022domain,
  title={Domain Knowledge-Informed Self-supervised Representations for Workout Form Assessment},
  author={Parmar, Paritosh and Gharat, Amol and Rhodin, Helge},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXVIII},
  pages={105--123},
  year={2022},
  organization={Springer}
}

@inproceedings{grauman2024ego,
  title={Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives},
  author={Grauman, Kristen and Westbury, Andrew and Torresani, Lorenzo and Kitani, Kris and Malik, Jitendra and Afouras, Triantafyllos and Ashutosh, Kumar and Baiyya, Vijay and Bansal, Siddhant and Boote, Bikram and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19383--19400},
  year={2024}
}

@inproceedings{gao2014jhu,
  title={Jhu-isi gesture and skill assessment working set (jigsaws): A surgical activity dataset for human motion modeling},
  author={Gao, Yixin and Vedula, S Swaroop and Reiley, Carol E and Ahmidi, Narges and Varadarajan, Balakrishnan and Lin, Henry C and Tao, Lingling and Zappella, Luca and B{\'e}jar, Benjam{\i}n and Yuh, David D and others},
  booktitle={MICCAI workshop: M2cai},
  volume={3},
  number={2014},
  pages={3},
  year={2014}
}
```


## Notes for this repo that we can remove for the final version
Notes

TODO for final:
- copy the final dataset to "jmhb0/ViddiffBench". Update the call in `data/load_viddiff_dataset.py` to load from "jmhb0/ViddiffBench".
- Put the '.py' files from data/ into the huggingface repo files. They are load_viddiff_dataset.py, download_data.py, but NOT update_dataset.py.
- There were some files in "viddiff/VidDiffBench" that were manually added ... the Readme, the .py scripts, the video files. 
- in download_data.py, change the "dataset_name" to "jmhb0/ViddiffBench".
- Remove the scripts for transforming the dataset  “update_dataset.py”
- run everything from scratch 
- pip install and so on 
- keepn the run `.sh` files. 


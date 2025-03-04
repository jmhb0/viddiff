#### decompose the proposer LLM into multiple stages:
# 1 differences only
# 2 subactions only
# 3 link the differences to the subactions


# system prompt bc I got some refusals to answer sports-related videosk
lookup_system_prompt_vision = {
    0: """\
You are a helpful vision assistant. 
The images here are using publicly available data from a machine learning dataset.
Your answers are been used to test your vision understanding. 
Your answers will not be used to take any actions in the real world.
"""
}

lookup_prompts_proposer_1_differences = {
    0: """\
I have two videos of an action with the following description: "{action}".

Propose a set of 'differences' in how this action may be performed between the two videos. 
For each difference, give a 'name', 'description', 'query_string', and 'num_frames'.

The 'name' is a very short name that can be used as a key.

The 'description' is a slightly longer description of the difference. 
The 'query_string' is the same as 'description'.
The  descriptions should be visual and about how the action is performed. For example 'the jump is higher', or 'the arm is more straight'.
The difference descriptions should not refer to a specific video. For example you would not say 'the jump in video B is higher'. 
Instead, the 'description' would be 'the jump is higher', and the statemet could be more true of one video or the other. 

Now suppose you had to judge whether the difference was stronger in one video, but you could only look at individual frames. 
	- What's the smallest number of frames you need, assuming the frame is well-chosen? Put the answer in 'num_frames'. The answer should be '1' or 'gt_1' (meaning 'greater than 1').
	- Once you have the frames to compare, create a 'query_string' that is a simple statement about this difference that is more true in one video vs the other video (based on the frames only). For example "the left leg is higher" or "movement is faster".

List {n_differences} differences.
Return a json like this, where the differences keys are stringified ints starting from 0.
{
	'0' : {
		"name" : "...",
		"description" : "...",
		"query_string" : "...",
		"num_frames": "1|gt_1",
	}, 
	...
}
"""
}


lookup_prompts_proposer_2_subactions = {
    0: """\
I have two videos of an action with the following description: "{action}".

Provide a 'stage transcript' as a list. These are sub-actions that make up that action.
Give 5 steps or fewer in the action transcript.

For each stage, give a 'name' for the stage, and a 'description' of that stage. 

For each stage, give a list of 'retrieval_strings'. 
These are strings that describe what is visible in the frame. 
Only describe the visual features. Only describe what is visible in a single frame. Focus on appearance. Focus on pose. Do not use the name of the action. Start each string with something similar to "A photo of a ...". 
Give at least {n_retrieval_keys} retrieval strings per stage.

Return a json like this:
{
	"stages" : [
		{ 
		"name : "",
		"description" : "...",
		"retrieval_strings" : ["A photo of a ...", ...],
		}, 
		...	
]}""",
}

# prompt to filter/refine the retrieval keys from `lookup_prompts_proposer_2_subactions`
# the retrieval keys proposed will tend to have low discriminability: they have overlap, or the string applies to multiple stages these check that
lookup_prompts_proposer_2_subactions_refiner = {
    0: """\
I have two videos of an action with the following description: "{action}".

Here is a 'stage transcript' of stages as a list with 'retrieval_strings' in each stage.
{stages}

The 'retrieval_strings' are strings that describe what is visible in the frame. They should describe visual features that are distinct to that stage. 
So, return the same stages and the same retrieval strings, but filtering out the strings that are too similar to strings in other stages. It's okay if a string is similar to strings in one or two other stages, but it should not be similar to strings in more stages.
The retrieval strings should only appear in the stages that are shown above. 

Return a json like this:
{
	"stages" : [
		"name : "",
		"description" : "...",
		"retrieval_strings" : ["A photo of a ...", ...],
	}, ...
]
}
"""
}


lookup_prompts_proposer_3_linking = {
    0: """\
I have two videos of an action with the following description: "{action}".

Here are a list of stages or subactions that make up that action:
{stages}

We also have differences in how this action may be performed between the two videos. 
The differences are specific statements that are more true in one video vs another. 
Here they are:
{differences}

Now we need to match each differences to a stage.
Return a list of the stages using their names.
If a difference is relevant to a particular stage, put its name in the 'difference' list. 
It's okay for a 'difference' to be visible in multiple stages. 
It's okay for some stages to have no difference.
Refer to stages and differences by their 'name' attribute. 

Return a json like this:
[	
	"<stage_name0>" : ["<difference_name0>", "<difference_name1>", ...],
	"<stage_name1>" : [],
	"<stage_name2>" : ["<difference_name1>", "<difference_name2>", ...],
	...
]
Please be careful. Every difference must appear at least once
""",
}


# here, we create separate query templates for single-image and multi-image cases.
lookup_prompts_differencing_1_frame = {
    0: """\
These images are from two videos of people performing an action with description: "{action}".
Which one shows more of the variation with this description: "{query_string}"?
(a) image 1, (b) image 2, (c) similar or can't tell.
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b|c"}""",
    # same as 0, but no 'c' option
    1: """\
These images are from two videos of people performing an action with description: "{action}".
Which one shows more of the variation with this description: "{query_string}"?
? (a) image 1, (b) image 2.
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b"}""",
    2: """\
These images are from two videos of people performing an action with description: "{action}".
Which one shows more of the variation with this description: "{query_string}"?
(a) image 1, (b) image 2, (c) similar.
Answer 'a' or 'b' only if confident. 
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b|c"}""",
    3: """\
These images are from two videos of people performing an action with description: "{action}".
Which one shows more of the variation with this description: "{query_string}"?
(a) image 1, (b) image 2, (c) similar or can't tell.
Please carefully compare these two images, and try to predict whether one image shows more on "{query_string}" than the other image.
If the differences are very minor, or they just looks similar/comparable, please choose c.
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b|c"}""",
    4: """\
These images are from two videos of people performing an action with description: "{action}".
Which one shows more of the variation with this description: "{query_string}"?
(a) image 1, (b) image 2.
The 'answer' must be 'a' or 'b'
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b"}""",
    5: """\
These images are from two videos of people performing an action with description: "{action}".
Which one shows more of the variation with this description: "{query_string}"?
(a) image 1, (b) image 2.
The 'answer' must be 'a' or 'b'
Answer in json:  {'answer':"a|b"}""",
}


lookup_prompts_differencing_multiframe = {
    0: """\
I have two videos of people performing an action with description: "{action}".
The first {num_frames} frames are from video A and the last {num_frames} frames are from video B.
For each video, the frames are very close together in the video: they are {time_diff} seconds apart.
Which one shows more of the variation with this description: "{query_string}"?
(a) video 1, (b) video 2, (c) similar or can't tell.
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b|c"}""",
    1: """\
I have two videos of people performing an action with description: "{action}".
The first {num_frames} frames are from video A and the last {num_frames} frames are from video B.
For each video, the frames are very close together in the video: they are {time_diff} seconds apart.
Which one shows more of the variation with this description: "{query_string}"?
(a) video 1, (b) video 2, (c) similar.
Answer 'a' or 'b' only if confident. 
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b|c"}""",
    # exact same as 0. Did this so that I could run experiments where the prompt key for single and multiframe differencing were matched.
    2: """\
I have two videos of people performing an action with description: "{action}".
The first {num_frames} frames are from video A and the last {num_frames} frames are from video B.
For each video, the frames are very close together in the video: they are {time_diff} seconds apart.
Which one shows more of the variation with this description: "{query_string}"?
(a) video 1, (b) video 2, (c) similar or can't tell.
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b|c"}""",
    3: """\
I have two videos of people performing an action with description: "{action}".
The first {num_frames} frames are from video A and the last {num_frames} frames are from video B.
For each video, the frames are very close together in the video: they are {time_diff} seconds apart.
Which one shows more of the variation with this description: "{query_string}"?
(a) video 1, (b) video 2, (c) similar.
Please carefully compare these frames, and try to predict whether one set of {num_frames} frames shows more on "{query_string}" than the other set of {num_frames} frames.
If the differences are very minor, or they just looks similar/comparable, please choose c.
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b|c"}""",
    4: """\
I have two videos of people performing an action with description: "{action}".
The first {num_frames} frames are from video A and the last {num_frames} frames are from video B.
For each video, the frames are very close together in the video: they are {time_diff} seconds apart.
Which one shows more of the variation with this description: "{query_string}"?
(a) video 1, (b) video 2
The 'answer' must be 'a' or 'b'
Answer in json:  {'answer_detailed' : "...", 'answer':"a|b"}""",
    5: """\
I have two videos of people performing an action with description: "{action}".
The first {num_frames} frames are from video A and the last {num_frames} frames are from video B.
For each video, the frames are very close together in the video: they are {time_diff} seconds apart.
Which one shows more of the variation with this description: "{query_string}"?
(a) video 1, (b) video 2
The 'answer' must be 'a' or 'b'
Answer in json:  {'answer':"a|b"}""",
}

prompt_template_mcq_ab_singlefrmae = """\
I have two videos of people performing an action with description: "{action}".
The first frame is from video a, and the second frame is from video b.

Here is a difference that describe how the action can be performed differently:
"{query_string}"

Your task is to predict whether the difference is more true for video 'a' or video 'b'.
If video 'a', then write "The answer is (a)".
If video 'b', then write "The answer is (b)".
"""
prompt_template_mcq_ab_multifrmae = """\
I have two videos of people performing an action with description: "{action}".
The first {num_frames} frames are from video A and the last {num_frames} frames are from video B.
For each video, the frames are very close together in the video: they are {time_diff} seconds apart.

Here is a difference that describe how the action can be performed differently:
"{query_string}"

Your task is to predict whether the difference is more true for video 'a' or video 'b'.
If video 'a', then write "The answer is (a)".
If video 'b', then write "The answer is (b)".
"""

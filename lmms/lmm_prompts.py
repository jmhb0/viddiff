## prompt templates for all eval modes
prompt_template_open = """\
Here are two videos of an action with the following description: "{action_description}".
{video_representation_description}

Return a list of 'differences' in how the action is being performed. 
Each difference should have a 'description' that is a specific statements that is more true in one video compared to the other video. 
Then there is a 'prediction' which is 'a' if the statement applies more to video a and 'b' if it applies more to video b.

The difference descriptions should be visual and about how the action is performed. 
For example 'the jump is higher', or 'the arm is more straight'.

The difference descriptions should not refer to a specific video. 
For example you would not say 'the jump in video B is higher'. 
Instead, the 'description' would be 'the jump is higher', and the 'prediction' is 'b'.

Suggest no more than {n_differences} differences.


Return a json like this, replacing '...' with actual content:
{
    "0" :  {
            "description" : "....",
            "prediction" : "a|b"
        },
    "1" :  {
            "description" : "....",
            "prediction" : "a|b"
        }, 
    ...
}
"""

# mode 1 
prompt_template_mcq_ab = """\
Here are two videos of an action with the following description: "{action_description}".
{video_representation_description}

Here is a difference that describe how the action can be performed differently:
{differences_annotated}

Your task is to predict whether the difference is more true for video 'a' or video 'b'.
If video 'a', then write "The answer is (a)".
If video 'b', then write "The answer is (b)".
"""

# mode 2 - closed, also a or b 
prompt_template_mode_2 = """\
Here are two videos of an action with the following description: "{action_description}".
{video_representation_description}


Below is a set of identified differences that describe how the action be performed differently.
Each difference is associated with a unique key:
{differences_annotated}

Your task is to predict, for each difference, whether it is more true for video 'a' or video 'b'. 
Return your predictions as a JSON object, using the keys above. For each difference, include the description and your prediction ("a" or "b"):
{target_out}
"""


prompt_template_mcq = """\
Here are two videos of an action with the following description: "{action_description}".
{video_representation_description}

We will provide a list of differences that describe how the action can be performed differently:
{differences_annotated}
Note that each difference has a key that should be used in the final prediction. 

Your task is to predict, for each difference, whether the statement is more true for video 'a' or 'b'. 

For example, if the difference '0' has prediction 'b' and difference '1' has prediction 'a', return this json:
{
    "0" : "b",
    "1" : "a",
}
"""

## prompt templates for explaining how the video is represented
video_rep_description_2_videos = """\
We have passed 'video a' and 'video b' as videos to the prompt in that order."""
# video_rep_description_1_video = """\
# We have passed in 1 video into the prompt, which is the concatenation of 'video a' and then 'video b'."""
video_rep_description_2_grids = None  # """We have passed in 1 video into the prompt, which is the concatenation of 'video a' and then 'video b'."""
video_rep_description_frames = """\
We have passed a sequence of images into the prompt. 
The first {vid0_nframes} are video a. The last {vid1_nframes} are video b.
The frame rate is the same and is {fps}.
"""
video_rep_description_2_grids = """\
We have passed two images into the prompt. 
The first image is a grid of images showing the frames of video a row-wise. 
The first image is a grid of images showing the frames of video b row-wise. 
The frame rate is the same.
"""
video_rep_description_2_videos = """\
We have passed 'video a' and 'video b' as videos to the prompt in that order."""
video_rep_description_first_frame="""\
We have provided two images. 
The first image is the first frame from 'video a'. 
The second image is the first frame from 'video b'.
Try to make make the best prediction possible given only one frame. 
"""

prompt_reformat_malformed_json = """\
Below I will give you the "LLM Output". That output should be close to a json, but with some issues. 
Your task is to reformat it to follow the correct JSON structure. 

The correct json structure is a dictionary, where the keys are are stringified ints. 
The values are a dict, with keys "description" and "prediction". 
The "description" is a string that is a short statement. 
The "prediction" is either the letter "a" or "b". 
{
    "0" : {
        "description" : "the jump is higher",
        "prediction" : "b",
    },
    "1" : {
        "description" : "the arm is more straight during the movement",
        "prediction" : "a",
    },
}

If there is text before or after the json, then ignore it. 
If the bad json is just "{}" then you can return an empty json. 

LLM OUTPUT: 
{llm_output}
"""




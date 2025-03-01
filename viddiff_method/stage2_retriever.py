"""

"""
import ipdb
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from pathlib import Path
import io
import logging
from line_profiler import LineProfiler
from datasets import Dataset
from PIL import Image

from apis import clip_utils
from viddiff_method import utils
from viddiff_method import utils_retriever


class Retriever():
    """
    Output of this is to make  dict `self.retrieved_frames`
    self.retrieved_frames[sample_idx] is a list with two dicts - one for each vid in the pair

    So for video 0 for example:
        self.retrieved_frames[sample_idx][0][variation_idx]
    Returns a list of frame numbers (int) retrieved for this query, e.g.
        retriever.retrieved_frames['0'][0]['5'] = [42, 43, 45] 

    """

    def __init__(self, args: dict, args_logging: dict, dataset: Dataset,
                 videos: list, proposals: dict):
        self.args = args
        self.dataset = dataset
        self.videos0 = videos[0]
        self.videos1 = videos[1]
        self.proposals = proposals
        self.verbose = args_logging.verbose

        self.results_subdir = Path(
            args_logging.results_dir) / "stage2_retriever"
        self.results_subdir.mkdir(exist_ok=True, parents=True)

    def retrieve_frames(self):
        # if self.args.eval_mode in (2, ):
        #     logging.info(f"eval mode {self.args.eval_mode}, getting gt frames")
        #     self.get_gt_frames()
        # else:
        self.get_clip_text_embeddings()
        self.get_clip_text_frame_similarity()
        self.do_temporal_segmentation()
        self.map_segmentation_to_frames()

        if self.args.do_random_retrieval:
            self.randomize_retrievals()

        return self.retrieved_frames

    def get_clip_text_embeddings(self):
        """
        Prepare all text embeddings to self.embeddings_text and save lookups from
        retrieval strings to embeddings in self.embeddings_action_text.


        `self.proposer` has `proposals` for each unique activity. 
        Each activity has a list of stages, and each stage has a list of 
        `retrieval_strings`. E.g. action i, stage j:
            proposer.proposals[i]['stages'][j]['retrieval_strings']

        Now lets concat all retrieval strings into one list 
            self.all_retlookup_video_to_retrieved_framesrieval_strings # (n_strings,)
        And save the CLIP text embeddings: 
            self.embeddings_text # (n_strings, d_clip)

        To map a particular stage's retrieval strings to the text embedding, 
        just record the idx to self.embeddings_text. E.g. E.g. action i, stage j:
            self.embeddings_action_text[i]['stages'][j] 
        Will be a list like [4,5,6].
        """
        self.embeddings_action_text = {}

        self.all_retrieval_strings = []
        text_idx = 0
        for sample_key, proposal in self.proposals.items():
            self.embeddings_action_text[sample_key] = {}
            self.embeddings_action_text[sample_key]["text_idxs"] = []
            self.embeddings_action_text[sample_key]["text_idxs_all"] = []
            stages = [p['name'] for p in proposal.stages]
            for j, stage in enumerate(stages):
                retrieval_strings = proposal.stages[j]['retrieval_strings']
                self.all_retrieval_strings.extend(retrieval_strings)
                num_strings = len(retrieval_strings)
                # save the idxs of the strings for this action
                text_idxs = list(range(text_idx, text_idx + num_strings))
                text_idx += num_strings
                self.embeddings_action_text[sample_key]["text_idxs"].append(
                    text_idxs)
                self.embeddings_action_text[sample_key][
                    "text_idxs_all"].extend(text_idxs)
                self.embeddings_action_text[sample_key]['stages'] = stages

        assert len(self.all_retrieval_strings) == text_idx

        # call CLIP text only once
        self.embeddings_text = clip_utils.get_embeddings(
            self.all_retrieval_strings, self.args.model_config.model, "text")

    def get_clip_text_frame_similarity(self):
        """
        For each pair of vids in self.dataset, get video embed, text embed,
        and cosine sim matrix for each action stage. 
        Create a new key as well.
        Save it to self.stages_retrieved
        """
        self.stages_retrieved = {}
        for sample, video0, video1 in zip(self.dataset, self.videos0,
                                          self.videos1):
            sample_key = sample['sample_key']
            video_pair = [video0, video1]
            # find activity info from the proposer
            stages = self.embeddings_action_text[sample_key]['stages']

            # do stuff for each video
            for i in [0, 1]:
                new_key = f"{sample_key}--{i}"
                res = {}

                # frame embeddings
                video = video_pair[i]['video']
                z_frames = clip_utils.get_embeddings_video(
                    video, model=self.args.model_config.model)
                assert len(z_frames) == len(video)

                res["stages"] = stages
                res['texts'] = []
                res['simmats'] = []

                # compute frame-to-text similarity matrix for each stage/subaction
                for j, stage in enumerate(stages):
                    text_idxs = self.embeddings_action_text[sample_key][
                        "text_idxs"][j]
                    texts = [self.all_retrieval_strings[t] for t in text_idxs]
                    z_text = self.embeddings_text[text_idxs]
                    simmat = cosine_similarity_matrix(z_text, z_frames)

                    res['texts'].append(texts)
                    res['simmats'].append(simmat)

                self.stages_retrieved[new_key] = res

    def do_temporal_segmentation(self):
        """ 
        Temporal segmentation: loop over all videos, and call self.segment_one_video. 
        Optionally calls image saving. 
        """
        for sample, video0, video1 in zip(self.dataset, self.videos0,
                                          self.videos1):
            sample_key = sample['sample_key']
            video_pair = [video0, video1]

            for i in [0, 1]:
                video = video_pair[i]

                key = f"{sample_key}--{i}"
                assert key in self.stages_retrieved.keys()
                simmats = self.stages_retrieved[key]['simmats']
                texts = self.stages_retrieved[key]['texts']
                stages = self.stages_retrieved[key]['stages']
                stage_segments = self.segment_one_video(
                    key, video, simmats, texts, stages)
                self.stages_retrieved[key]['stage_segments'] = stage_segments
                if self.args.log_imgs:
                    logging.info("Saving images")
                    self._save_similarity_img(key, video['video'], simmats,
                                              texts, stages)
                    self._save_segmentation_img(key, video['video'], stages,
                                                stage_segments)
                    # this next one is extra slow
                    self._save_frame_sets(key, video['video'], stages,
                                          stage_segments)

    def segment_one_video(self, key, video, simmats, texts, stages):
        """ 

        Modes:
        0: fix each stage length to 1.3* 1/n_stages. Choose the segment with the 
        highest mean clip score, using the first clip retrieval key.
        1: very basic break up the video exactly evenly into 1/n_stages. Then 
        also grow them by 30%
        2: first iteration of Viterbi decoding: just take the CLIP scores from 
        the first retrieval key 

        ranges [start,end), like normal python indexing (not inclusive on RHS)
        """

        if self.args.mode == 0:
            stage_frames = []

            for stage, text_set, simmat in zip(stages, texts, simmats):
                stage_length = int(1.3 * (len(video) / len(stages)))
                if stage_length % 2 == 0:
                    stage_length += 1

                simmat_norm = normalize_matrix_rows(simmat.copy())
                scores = simmat_norm[0]  # take the top row

                kernel = stage_length
                scores_conv = convolution_1d_mean(scores, kernel_size=kernel)
                stage_midframe = np.argmax(scores_conv)

                min_choice = kernel // 2
                max_choice = len(scores_conv) - kernel // 2 - 1
                stage_midframe = max(stage_midframe, min_choice)
                stage_midframe = min(stage_midframe, max_choice)

                start = stage_midframe - stage_length // 2
                end = stage_midframe + stage_length // 2 + 1

                stage_frames.append([start, end])

        elif self.args.mode == 1:
            n_frames = len(video)
            n_stages = len(stages)
            stage_size, remainder = divmod(n_frames, n_stages)

            stage_frames = []
            start = 0
            for i in range(n_stages):
                end = start + stage_size + (1 if i < remainder else 0)
                stage_frames.append([start, end])  #  exclusive range
                start = end

            # give some extra range +/- 30% (or 15% each size)
            grow_seg_each_side = max(1, int(0.15 * stage_size))
            stage_frames[0][1] += 1  # grow first stage on right
            stage_frames[-1][0] -= 1  # grow second stage on left
            for i in range(1, n_stages - 1):
                stage_frames[i][0] -= 1
                stage_frames[i][1] += 1

        elif self.args.mode == 2:
            # choose just the first
            nstages = len(stages)
            scores = np.stack([s.mean(axis=0) for s in simmats]).copy()
            scores = normalize_matrix_rows(scores)
            mask = scores == 0
            scores[mask] = 1e-5
            probs = scores
            class_preds = utils_retriever.run_viterbi(probs)

            # visualize the probs
            if self.args.log_imgs:
                self._save_viterbi_logprobs_img(probs, key)

            stage_frames = []
            for i in range(nstages):
                idxs = np.where(np.array(class_preds) == i)[0]
                stage_frames.append([idxs.min(), idxs.max() + 1])

        else:
            raise NotImplementedError()

        assert len(stage_frames) == len(stages)

        return stage_frames

    def _save_viterbi_logprobs_img(self, probs, key):
        """ 
        save plot of the matrix that will go into the Viterbi algorithm
        """
        fig_probs, ax = plt.subplots()
        cax = ax.imshow(probs, cmap='gray_r')
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("bottom", size="20%", pad=0.3)
        plt.colorbar(cax, cax=cax2, pad=0.0, orientation='horizontal')
        f_stem = self.results_subdir / key
        fig_probs.savefig(f"{f_stem}_viterbi_input_probs.png")
        plt.close()

    def _save_similarity_img(self, key, video, simmats, texts, stages):
        """
        Make plots.
        """
        # this part is for visualization
        imgs_plot = []
        grid = utils.create_image_grid_with_labels(video, nrow=5)
        w, h = grid.size
        grid = grid.resize((w // 6, h // 6))  # because the grid image is big

        # for each stage in the action transcript
        imgs_clip = []
        for stage, text_set, simmat in zip(stages, texts, simmats):
            f_simmat, img_simmat = plot_matrix(simmat, norm_rows=False)
            f_simmat_norm, img_simmat_norm = plot_matrix(simmat,
                                                         norm_rows=True)
            img_texts = utils.print_strings_on_image([f"Stage: {stage}"] +
                                                     text_set,
                                                     numbering_start_idx=-1)
            # now stack the images together - grid, 2 simmats, text
            img = utils.stack_images(img_simmat,
                                     img_simmat_norm,
                                     mode='v',
                                     resize_width=True,
                                     resize_height=True)
            img = utils.stack_images(img_texts,
                                     img,
                                     mode='h',
                                     resize_height=True)
            img = utils.add_border_to_img(img, "top")
            imgs_clip.append(img)

        img_super = imgs_clip[0]
        for img in imgs_clip[1:]:
            img_super = utils.stack_images(img_super,
                                           img,
                                           mode='v',
                                           resize_width=False,
                                           resize_height=False)

        f_stem = self.results_subdir / key
        img_super.save(f"{f_stem}clipscores.png")
        grid.save(f"{f_stem}grid.png")

        plt.close()

    def _save_segmentation_img(self, key, video, stagenames, stage_segments):
        """ 
        Save the temporal action transcript 
        """
        # temporal segmentation map
        n_stages, n_frames = len(stagenames), len(video)
        matrix = np.zeros((n_stages, n_frames), dtype=int)
        for i in range(n_stages):
            slc = slice(*stage_segments[i])
            matrix[i, slc] = 1

        fig, fig_img = plot_matrix(matrix,
                                   colorbar=False,
                                   gridlines=True,
                                   yticklabels=stagenames)
        f_stem = self.results_subdir / key
        fig_img.save(f"{f_stem}transcript.png")

    def _save_frame_sets(self,
                         key,
                         video,
                         stages,
                         stage_segments,
                         downscale=3):
        """ """
        results_frames = self.results_subdir / f"{key}_frame_preds"
        results_frames.mkdir(exist_ok=True)
        for i, (name, segs) in enumerate(zip(stages, stage_segments)):
            frames = video[slice(segs[0], segs[1])]
            frames = frames[:, ::downscale, ::downscale]

            grid = utils.create_image_grid_with_labels(frames,
                                                       nrow=4,
                                                       title=f"{i}_{name}",
                                                       title_size=200)
            fname = results_frames / f"seg_{i}_{name}.png"
            grid.save(fname)

    def map_segmentation_to_frames(self):
        """ """

        self.retrieved_frames = {}

        for sample, video0, video1 in zip(self.dataset, self.videos0,
                                          self.videos1):
            sample_key = sample['sample_key']
            video_pair = [video0, video1]
            self.retrieved_frames[sample_key] = []

            for video_idx in [0, 1]:

                # temporal segmentation info
                video_key = f"{sample_key}--{video_idx}"
                video_stages_retrieved = self.stages_retrieved[video_key]
                fps = video_pair[video_idx]['fps']

                # get the proposer, which has the variation and variation->stage mapping info
                proposal = self.proposals[sample_key]

                # get the final frame retrieval predictions
                retrieval_frames = self._map_seg_to_frames_one_video(
                    video_stages_retrieved, proposal.differences,
                    proposal.lookup_diffidx_to_stages, fps, sample_key)

                self.retrieved_frames[sample_key].append(retrieval_frames)

    def _map_seg_to_frames_one_video(self,
                                     video_stages_retrieved,
                                     differences,
                                     lookup_diffidx_to_stages,
                                     fps,
                                     sample_key=None):
        """ 
        For one video, use temporal segmentation info and difference info to
        get the final retrieved frames
        """
        # standard flow
        frame_idxs = {}
        stage_names_to_idx = {
            name: i
            for i, name in enumerate(video_stages_retrieved['stages'])
        }
        # get temporal act segmentation. The saved frames are exclusive indexing,
        # e.g. slice(0,4) goes up to idx 3. So subtract the end by 1, since we want the end frame.
        stage_segments_retrieved = video_stages_retrieved['stage_segments']
        for i in range(len(stage_segments_retrieved)):
            stage_segments_retrieved[i][
                -1] = stage_segments_retrieved[i][-1] - 1

        # get the frame_idxs per difference
        for diff_idx, diff in differences.items():
            num_frames = diff['num_frames']
            assert num_frames in ('1', 'gt_1')
            stages = lookup_diffidx_to_stages[diff_idx]
            # handle special case of no match
            if len(stages) != 0:
                stages_names = [s['name'] for s in stages]
                stages_idx = [stage_names_to_idx[n] for n in stages_names]
            else:
                stages_idx = [1]

            if num_frames == "1":
                # for now, just take the first stage
                stage_idx = stages_idx[0]
                # start and end frames for the recovered stage segment
                stage_segments = video_stages_retrieved['stage_segments'][
                    stage_idx]

                frame_idx = int(np.mean(stage_segments))  # implicit rounding

                frame_idxs[diff_idx] = [
                    frame_idx,
                ]

            elif num_frames == "gt_1":
                stage_segments = [
                    stage_segments_retrieved[i] for i in stages_idx
                ]
                stage_segments_flat = [
                    i for sub in stage_segments for i in sub
                ]

                ## todo: how to handle special case of no matches?
                if len(stage_segments_flat) == 0:
                    # for now, make the stage_seg_flat be first and last frame which will make the mid-frame the mid of the whole video
                    stage_segments_flat = [
                        video_stages_retrieved['stage_segments'][0][0],
                        video_stages_retrieved['stage_segments'][-1][1]
                    ]

                midframe = int(
                    (max(stage_segments_flat) + min(stage_segments_flat)) / 2)
                assert self.args.multiframe.nframes % 2 == 1

                # similar to in the preprocess.py
                frames_sep = round(self.args.multiframe.frames_sep_seconds *
                                   fps)
                mid_index = self.args.multiframe.nframes // 2
                frames = []
                for i in range(self.args.multiframe.nframes):
                    frame = midframe + (i - mid_index) * frames_sep
                    frames.append(frame)

                # if frames are outside the video range, then shift it
                min_frame = 0
                max_frame = video_stages_retrieved['simmats'][0].shape[1] - 1

                if max(frames) > max_frame:
                    shift_left = max(frames) - max_frame
                    frames = [f - shift_left for f in frames]

                if min(frames) < min_frame:
                    shift_right = min_frame - min(frames)
                    frames = [f + shift_right for f in frames]

                # save
                frame_idxs[diff_idx] = frames

            else:
                raise ValueError

        return frame_idxs

    def get_gt_frames(self):
        """ 
        Copy the retrieval frames that are already in the dataset 
        """

        self.retrieved_frames = {}

        for sample_key, val in self.dataset['samples'].items():
            self.retrieved_frames[sample_key] = [
                val['videos'][0]['retrieval_frames'],
                val['videos'][1]['retrieval_frames'],
            ]

    def randomize_retrievals(self):
        import random
        random_seed = self.args.seed
        for sample_key, frame_idxs in self.retrieved_frames.items():
            for vid_idx in [0, 1]:
                nframes_vid = len(self.dataset['samples'][sample_key]['videos']
                                  [vid_idx]['video'])
                for k, v in self.retrieved_frames[sample_key][vid_idx].items():

                    nframe_idxs = len(v)
                    if nframe_idxs == 1:
                        self.retrieved_frames[sample_key][vid_idx][k] = [
                            random.randint(0, nframes_vid - 1)
                        ]

                    else:
                        assert nframe_idxs == 3, 'next few lines assumes this is true'
                        sep = v[1] - v[0]
                        mid = random.randint(sep, nframes_vid - sep - 1)
                        self.retrieved_frames[sample_key][vid_idx][k] = [
                            mid - sep, mid, mid + sep
                        ]


def normalize_matrix_rows(matrix):
    min_vals = matrix.min(axis=1, keepdims=True)
    max_vals = matrix.max(axis=1, keepdims=True)
    matrix = (matrix - min_vals) / np.where(max_vals - min_vals == 0, 1,
                                            max_vals - min_vals)
    return matrix


def convolution_1d_mean(array, kernel_size):
    """
    Apply 1D mean convolution over a 1D array.

    Parameters:
    array (list or np.ndarray): The input 1D array.
    kernel_size (int): The size of the kernel (must be a positive odd integer).

    Returns:
    np.ndarray: The resulting array after applying mean convolution.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer.")

    # Convert input to NumPy array for convenience
    array = np.array(array)
    # Compute the padding size
    pad_size = kernel_size // 2
    # Pad the array with the 'edge' mode to handle boundaries
    padded_array = np.pad(array, pad_size, mode='edge')
    # Initialize the result array
    result = np.empty(array.shape)

    # Perform the convolution
    for i in range(len(array)):
        # Compute the mean for the current window
        result[i] = padded_array[i:i + kernel_size].sum()

    return result


def cosine_similarity_matrix(x, y):
    assert x.ndim == 2 and y.ndim == 2
    # Normalize the embeddings to have unit length
    x_normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_normalized = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute the cosine similarity
    similarity_matrix = np.dot(x_normalized, y_normalized.T)

    return similarity_matrix


def plot_matrix(matrix,
                norm_rows=False,
                colorbar=True,
                cmap='gray_r',
                gridlines=False,
                yticklabels=None,
                xtick_steps=3):
    """ 
    Used for plotting similarity matrices and action transcripts.
    The x-axis is video frames, and is usually longer. 

    Default settings
        yticks will be numbered (0,matrix.shape[0],1). 
        yticklabels are the same unless specified by yticklabels 
        xticks will be numbered (0,matrix.shape[1],xtick_steps) with xtick_steps
            equal to 3. 

        colorbar put at the bottom 
    """
    if norm_rows:
        matrix = matrix.copy()
        matrix = normalize_matrix_rows(matrix)

    # plot the matrix
    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap=cmap)

    # ticks and labels
    nums_x = list(range(0, matrix.shape[1], xtick_steps))
    nums_y = list(range(matrix.shape[0]))
    ax.set(xticks=nums_x,
           yticks=nums_y,
           xticklabels=nums_x,
           yticklabels=nums_y)

    if yticklabels is not None:
        assert len(yticklabels) == len(nums_y)
        ax.set(yticklabels=yticklabels)
        default_font_size = mpl.rcParams['font.size']
        plt.gca().set_yticklabels(plt.gca().get_yticklabels(),
                                  fontsize=default_font_size / 2)

    if gridlines:
        plt.grid(which='both', color='g', linestyle='-', linewidth=0.3)

    if colorbar:
        # this part is to make the colorbar smaller
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("bottom", size="20%", pad=0.3)
        plt.colorbar(cax, cax=cax2, pad=0.0, orientation='horizontal')

    plt.tight_layout(pad=0)

    # save as a png to buffer - so we can use the savefig options for tight layout
    buf = io.BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0, format='png')
    buf.seek(0)
    fig_img = Image.open(buf)

    plt.close()
    return fig, fig_img

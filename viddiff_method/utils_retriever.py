"""
Vitrbi decoding funcs from Anna Kukleva Aug 2018 https://github.com/Annusha/unsup_temp_embed/tree/master .
Copied code is:
    classes Viterbi, Grammar
    funcs create_grammar, bounds, plot_segm

The other funcs are by JB
"""

import ipdb
import numpy as np
import logging
import matplotlib.pyplot as plt


class Grammar:
    def __init__(self, states):
        """
        Args:
            states: flat sequence (list) of states (class State)
        """
        self._states = states
        self._framewise_states = []
        self.name = "%d" % len(states)

    def framewise_states(self):
        return_states = list(map(lambda x: self._states[x], self._framewise_states))
        return return_states

    def reverse(self):
        self._framewise_states = list(reversed(self._framewise_states))

    def __getitem__(self, idx):
        return self._framewise_states[idx]

    def set_framewise_state(self, states, last=False):
        """Set states for each item in a sequence.
        Backward pass by setting a particular state for computed probabilities.
        Args:
            states: either state indexes of the previous step
            last: if it the last item or not

        Returns:

        """
        if not last:
            state = int(states[[self._framewise_states[-1]]])
        else:
            # state = int(self._states[-1])
            state = int(len(self._states) - 1)

        self._framewise_states.append(state)

    def states(self):
        return self._states

    def __len__(self):
        return len(self._states)


def create_grammar(n_states):
    """Create grammar out of given number of possible states with 1 sub-action
    each.
    """
    states = list(range(n_states))
    grammar = Grammar(states)
    return grammar


class Viterbi:
    def __init__(self, grammar, probs, transition=0.5):
        self._grammar = grammar
        self._transition_self = -np.log(transition)
        self._transition_next = -np.log(1 - transition)
        self._transitions = np.array([self._transition_self, self._transition_next])
        # self._transitions = np.array([0, 0])

        self._state = []

        self._probs = probs
        self._state = self._probs[0, 0]
        self._number_frames = self._probs.shape[0]

        # probabilities matrix
        self._T1 = np.zeros((len(self._grammar), self._number_frames)) + np.inf
        self._T1[0, 0] = self._state
        # argmax matrix
        self._T2 = np.zeros((len(self._grammar), self._number_frames)) + np.inf
        self._T2[0, 0] = 0

        self._frame_idx = 1

    def inference(self):
        while self._frame_idx < self._number_frames:
            for state_idx, state in enumerate(self._grammar.states()):
                idxs = np.array([max(state_idx - 1, 0), state_idx])
                probs = (
                    self._T1[idxs, self._frame_idx - 1]
                    + self._transitions[idxs - max(state_idx - 1, 0)]
                    + self.get_prob(state)
                )
                self._T1[state_idx, self._frame_idx] = np.min(probs)
                self._T2[state_idx, self._frame_idx] = np.argmin(probs) + max(
                    state_idx - 1, 0
                )
            self._frame_idx += 1

    def get_prob(self, state):
        return self._probs[self._frame_idx, state]

    def backward(self, strict=True):
        if strict:
            last_state = (
                -1 if self._T2.shape[0] < self._T2.shape[1] else self._T2.shape[1]
            )
            self._grammar.set_framewise_state(self._T2[last_state, -1], last=True)
        else:
            self._grammar.set_framewise_state(self._T1[..., -1], last=True)

        for i in range(self._T1.shape[1] - 1, 0, -1):
            self._grammar.set_framewise_state(self._T2[..., i])
        self._grammar.reverse()

    def loglikelyhood(self):
        return -1 * self._T1[-1, -1]

    def alignment(self):
        return self._grammar.framewise_states()

    def calc(self, alignment):
        self._sum = np.sum(
            np.abs(self._probs[np.arange(self._number_frames), alignment])
        )


def plot_segm(segmentation, colors, name=""):
    # mpl.style.use('classic')
    fig = plt.figure(figsize=(16, 4))
    plt.axis("off")
    plt.title(name, fontsize=20)
    # plt.subplots_adjust(top=0.9, hspace=0.6)
    gt_segm, _ = segmentation["gt"]
    ax_idx = 1
    plots_number = len(segmentation)
    ax = fig.add_subplot(plots_number, 1, ax_idx)
    ax.set_ylabel(
        "GT", fontsize=30, rotation=0, labelpad=40, verticalalignment="center"
    )
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # make_axes_area_auto_adjustable(ax)
    # plt.title('gt', fontsize=20)
    v_len = len(gt_segm)
    for start, end, label in bounds(gt_segm):
        ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1.0)
    for key, (segm, label2gt) in segmentation.items():
        if key in ["gt", "cl"]:
            continue
        ax_idx += 1
        ax = fig.add_subplot(plots_number, 1, ax_idx)
        ax.set_ylabel(
            "OUTPUT", fontsize=30, rotation=0, labelpad=60, verticalalignment="center"
        )
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # make_axes_area_auto_adjustable(ax)
        segm = list(map(lambda x: label2gt[x], segm))
        for start, end, label in bounds(segm):
            ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1.0)

    return fig
    # fig.savefig(path, transparent=False)


def bounds(segm):
    start_label = segm[0]
    start_idx = 0
    idx = 0
    while idx < len(segm):
        try:
            while start_label == segm[idx]:
                idx += 1
        except IndexError:
            yield start_idx, idx, start_label
            break

        yield start_idx, idx, start_label
        start_idx = idx
        start_label = segm[start_idx]


def run_viterbi(probs):
    """
    input shape (nclasses, nframes)
    """
    probs = probs.transpose()

    nframes, nclasses = probs.shape

    assert probs.min() >= 0 and probs.max() <= 1
    if nclasses >= nframes:
        logging.warning(
            f"nclasses ({nclasses}) >= nframes ({nframes}), returning range(nframes) as temporal segmentation"
        )
        return list(range(nframes))

    likelihood_grid = np.log(probs)
    pi = list(range(nclasses))
    grammar = Grammar(pi)
    viterbi = Viterbi(grammar=grammar, probs=(-1 * likelihood_grid))
    viterbi.inference()
    viterbi.backward(strict=True)
    pred_classes = np.ones(nframes, dtype=int) * -1
    pred_classes = viterbi.alignment()
    return pred_classes

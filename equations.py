import numpy as np
import itertools


def _calc_row_of_sampling(T, row, steps):
    V = np.zeros(T.shape)
    V[range(len(T)),range(len(T))] = 1
    not_probs = 1
    for step in range(steps):
        V = np.matmul(V, T)
        not_probs *= (1-V)
        V[:,row] = 0
        V /= V.sum(axis=1)[:,None]
    row_prob = (1-not_probs)[:,row]
    return row_prob


def _calc_discover_probs_stepping(T, steps=1, self_known=True):
    n_states = len(T)
    discover_probs = []
    for state in range(n_states):
        discover_probs.append(
            _calc_row_of_sampling(T, state, steps))
    discover_probs = np.array(discover_probs).T
    if self_known is True:
        discover_probs[range(n_states), range(n_states)] = 1
    return discover_probs


def calc_discover_probs(T, steps=1, clones=1, self_known=True):
    """The probability of discovering state j after S steps from state i
       with M trajectories."""
    discover_probs = _calc_discover_probs_stepping(
        T, steps=steps, self_known=self_known)
    discover_probs = np.array(1-((1-discover_probs)**clones))
    return discover_probs

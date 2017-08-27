import numpy as np


def _evens_select_states(msm, n_clones):
    # determine discovered states from msm
    counts_per_state = np.array(msm.tcounts_.sum(axis=1)).flatten()
    unique_states = np.where(counts_per_state > 0)[0]
    # calculate the number of clones per state and the balance to
    # match n_clones
    clones_per_state = int(n_clones / len(unique_states))
    remainder_states = n_clones % len(unique_states)
    # generate states to simulate list
    repeat_states_to_simulate = np.repeat(unique_states, clones_per_state)
    remainder_states_to_simulate = np.random.choice(
        unique_states, remainder_states, replace=False)
    total_states_to_simulate = np.concatenate(
        [repeat_states_to_simulate, remainder_states_to_simulate])
    return total_states_to_simulate


def __unbias_state_selection(states, rankings, n_selections, select_max=True):
    # determine the unique ranking values
    unique_rankings = np.unique(rankings)
    # sort states and rankings
    iis_sort = np.argsort(rankings)
    sorted_rankings = rankings[iis_sort]
    sorted_states = states[iis_sort]
    # if we're maximizing the rankings, reverse the sorts
    if select_max:
        unique_rankings = unique_rankings[::-1]
        sorted_rankings = rankings[::-1]
        sorted_states = states[::-1]
    # shuffle states with unique rankings to unbias selections
    # this is done upto n_selections
    tot_state_count = 0
    for ranking_num in unique_rankings:
        iis = np.where(sorted_rankings == ranking_num)
        sorted_states[iis] = sorted_states[np.random.permutation(iis)]
        tot_state_count += len(iis)
        if tot_state_count > n_selections:
            break
    return sorted_states[:n_selections]


class evens:
    def __init__(self):
        pass

    def select_states(self, msm, n_clones):
        return _evens_select_states(msm, n_clones)


class min_counts:
    def __init__(self):
        pass

    def select_states(self, msm, n_clones):
        # determine discovered states from msm
        counts_per_state = np.array(msm.tcounts_.sum(axis=1)).flatten()
        unique_states = np.where(counts_per_state > 0)[0]
        # if not enough discovered states for selection of n_clones,
        # selects states using the evens method
        if len(unique_states) < n_clones:
            states_to_simulate = _evens_select_states(msm, n_clones)
        # selects the n_clones with minimum counts
        else:
            counts_rankings = counts_per_states[unique_states]
            states_to_simulate = __unbias_state_selection(
                unique_states, counts_rankings, n_clones, select_max=False)
        return states_to_simulate

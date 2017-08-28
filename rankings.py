import numpy as np
from . import scalings

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


def _unbias_state_selection(states, rankings, n_selections, select_max=True):
    # determine the unique ranking values
    unique_rankings = np.unique(rankings)
    # sort states and rankings
    iis_sort = np.argsort(rankings)
    sorted_rankings = rankings[iis_sort]
    sorted_states = states[iis_sort]
    # if we're maximizing the rankings, reverse the sorts
    if select_max:
        unique_rankings = unique_rankings[::-1]
        sorted_rankings = sorted_rankings[::-1]
        sorted_states = sorted_states[::-1]
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


def counts_ranking(msm):
    counts_per_state = np.array(msm.tcounts_.sum(axis=1)).flatten()
    unique_states = np.where(counts_per_state > 0)[0]
    return unique_states, counts_per_state[unique_states]


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
        unique_states, counts_rankings = counts_ranking(msm)
        # if not enough discovered states for selection of n_clones,
        # selects states using the evens method
        if len(unique_states) < n_clones:
            states_to_simulate = _evens_select_states(msm, n_clones)
        # selects the n_clones with minimum counts
        else:
            states_to_simulate = _unbias_state_selection(
                unique_states, counts_rankings, n_clones, select_max=False)
        return states_to_simulate


class FAST:
    def __init__(
            self, state_rankings,
            directed_scaling = scalings.feature_scale(maximize=True),
            statistical_component = counts_ranking,
            statistical_scaling = scalings.feature_scale(maximize=False),
            alpha = 1, alpha_percent=False):
        self.state_rankings = state_rankings
        self.directed_scaling = directed_scaling
        self.statistical_component = statistical_component
        self.statistical_scaling = statistical_scaling
        self.alpha = alpha
        self.alpha_percent = alpha_percent
        if self.alpha_percent and ((self.alpha < 0) or (self.alpha > 1)):
            raise


    def select_states(self, msm, n_clones):
        unique_states, statistical_ranking = self.statistical_component(msm)
        if len(unique_states) < n_clones:
            states_to_simulate = _evens_select_states(msm, n_clones)
        # selects the n_clones with minimum counts
        else:
            directed_weights = self.directed_scaling.scale(
                self.state_rankings[unique_states])
            statistical_weights = self.statistical_scaling.scale(
                statistical_ranking)
            if self.alpha_percent:
                total_rankings = (1-self.alpha)*directed_weights + \
                    self.alpha*statistical_weights
            else:
                total_rankings = directed_weights + self.alpha*statistical_weights
            states_to_simulate = _unbias_state_selection(
                unique_states, total_rankings, n_clones, select_max=True)
        return states_to_simulate

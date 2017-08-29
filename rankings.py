import numpy as np
import time
import scipy.sparse as spar
from . import scalings


########################################################################
#                           helper functions                           #
########################################################################


def _evens_select_states(msm, n_clones):
    """Helper function for evens state selection. Picks among all
    discovered states evenly. If more states were discovered than
    clones, randomly picks remainder states."""
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
    """Unbiases state selection due to state labeling. Shuffles states
    with equivalent rankings so that state index does not influence
    selection probability."""
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


def get_unique_states(msm):
    """returns a list of the visited states within an msm object"""
    tcounts = msm.tcounts_
    unique_states = np.unique(np.nonzero(tcounts)[0])
    return unique_states


########################################################################
#                            page rankings                             #
########################################################################


def generate_aij(tcounts, spreading=False):
    """Generates the adjacency matrix used for page ranking.

    Parameters
    ----------
    tcounts : matrix, shape=(n_states, n_states)
        The count matrix of an MSM. Can be dense or sparse.
    spreading : bool, default=False
        Optionally transposes matrix to do counts spreading instead of
        page rank.

    Returns
    ----------
    aij : matrix, shape=(n_states, n_states)
        The adjacency matrix used for page ranking.

    """
    # check if sparse matrix
    if not spar.isspmatrix(tcounts):
        iis = np.where(tcounts != 0)
        tcounts = spar.coo_matrix(
            (tcounts[iis], iis), shape=(len(tcounts), len(tcounts)))
    else:
        tcounts = tcounts.tocoo()
    # get row and col information
    row_info = tcounts.row
    col_info = tcounts.col
    tcounts.data = np.zeros(len(tcounts.data)) + 1
    tcounts.setdiag(0) # Sets diagonal to zero
    tcounts = tcounts.tocsr()
    tcounts.eliminate_zeros()
    # optionally does spreading
    if spreading:
        tcounts = (tcounts + tcounts.T) / 2.
        aij = normalize(tcounts, norm='l1', axis=1)
    else:
        # Set 0 connections to 1 (
        # this doesn't change anything and avoids dividing by zero)
        connections = tcounts.sum(axis = 1)
        iis = np.where(connections == 0)
        connections[iis] = 1
        # Convert 1/Connections to sparse matrix format
        connections = spar.coo_matrix(connections).data
        con_len = len(connections)
        iis = (np.array(range(con_len)), np.array(range(con_len)))
        inv_connections = spar.coo_matrix(
            (connections**-1, iis), shape=(con_len, con_len))
        aij = tcounts.transpose() * inv_connections
    return aij

def rank_aij(aij, d=0.85, Pi=None, max_iters=100000, norm=True):
    """Ranks the adjacency matrix.
    
    Parameters
    ----------
    aij : matrix
        The adjacency matrix used for ranking.
    d : float
        The weight of page ranks [0, 1]. A value of 1 is pure page rank
        and 0 is all the initial ranks.
    Pi : array, default=None
        The prior ranks.
    max_iters : int, default=100000
        The maximum number of iterations to check for convergence.
    norm : bool, default=True
        Normilizes output ranks

    Returns
    ----------
    The rankings of each state

    """
    # if Pi is None, set it to 1/total states
    if Pi is None:
        Pi = np.zeros(int(N))
        Pi[:] = 1/N
    # set error for page ranks
    error = 1 / N**5
    # first pass of rankings
    new_page_rank = (1 - d) * Pi + d * aij.dot(Pi)
    pr_error = np.sum(np.abs(Pi - new_page_rank))
    page_rank = new_page_rank
    # iterate until error is below threshold
    iters = 0
    while pr_error > error:
        new_page_rank = (1 - d) * Pi + d * aij.dot(page_rank)
        pr_error = np.sum(np.abs(page_rank - new_page_rank))
        page_rank = new_page_rank
        iters += 1
        # error out if does not converge
        if iters > max_iters:
            raise
    # normalize rankings
    if norm:
        page_rank *= 100./page_rank.sum()
    return page_rank


########################################################################
#                           ranking classes                            #
########################################################################


class base_ranking:
    """base ranking class. Pieces out selection of states from
    independent rankings"""

    def __init__(self, maximize_ranking=True):
        self.maximize_ranking = maximize_ranking

    def select_states(self, msm, n_clones):
        # determine discovered states from msm
        unique_states = get_unique_states(msm)
        # if not enough discovered states for selection of n_clones,
        # selects states using the evens method
        if len(unique_states) < n_clones:
            states_to_simulate = _evens_select_states(msm, n_clones)
        # selects the n_clones with minimum counts
        else:
            rankings = self.rank(msm, unique_states=unique_states)
            states_to_simulate = _unbias_state_selection(
                unique_states, rankings, n_clones,
                select_max=self.maximize_ranking)
        return states_to_simulate


class page_ranking(base_ranking):
    """page ranking. ri = (1-d)*init_ranks + d*aij"""

    def __init__(
            self, d, init_pops=True, max_iters=100000, norm=True,
            spreading=False, maximize_ranking=True):
        """
        Parameters
        ----------
        d : float
            The weight of page ranks [0, 1]. A value of 1 is pure page rank
            and 0 is all the initial ranks.
        init_pops : bool, default=True
            Optionally uses the populations within an MSM as the initial ranks.
        max_iters : int, default=100000
            The maximum number of iterations to check for convergence.
        norm : bool, default=True
            Normilizes output ranks
        spreading : bool, default = False
            Solves for page ranks with the transpose of aij.
        """
        self.d = d
        self.init_pops = init_pops
        self.max_iters = max_iters
        self.norm = norm
        self.spreading = spreading
        base_ranking.__init__(self, maximize_ranking=maximize_ranking)

    def rank(self, msm, unique_states=None):
        # generate aij matrix
        if unique_states is None:
            unique_states = get_unique_states(msm)
        tcounts = msm.tcounts_
        tcounts_sub = tcounts.tocsr()[:, unique_states][unique_states, :]
        aij = generate_aij(tcounts_sub)
        # determine the initial ranks
        N = float(aij.shape[0])
        if self.init_pops:
            Pi = msm.eq_probs_[unique_states]
        else:
            Pi = None
        rankings = rank_aij(
            aij, d=self.d, Pi=Pi, max_iters=self.max_iters, norm=self.norm)
        return rankings


class evens:
    """Evens ranking object"""

    def __init__(self):
        pass

    def select_states(self, msm, n_clones):
        return _evens_select_states(msm, n_clones)


class counts(base_ranking):
    """Min-counts ranking object. Ranks states based on their raw
    counts."""

    def __init__(self, maximize_ranking=False):
        base_ranking.__init__(self, maximize_ranking=maximize_ranking)
    
    def rank(self, msm, unique_states=None):
        counts_per_state = np.array(msm.tcounts_.sum(axis=1)).flatten()
        if unique_states is None:
            unique_states = np.where(counts_per_state > 0)[0]
        return counts_per_state[unique_states]


class FAST(base_ranking):
    """FAST ranking object"""

    def __init__(
            self, state_rankings,
            directed_scaling = scalings.feature_scale(maximize=True),
            statistical_component = counts(),
            statistical_scaling = scalings.feature_scale(maximize=False),
            alpha = 1, alpha_percent=False, maximize_ranking=True):
        """
        Parameters
        ----------
        state_rankings : array, shape = (n_states, )
            An array with ranking values for each state. This is
            previously calculated for each state in the MSM.
        directed_scaling : scaling object, default = feature_scale(maximize=True)
            An object used for scaling the directed component values.
        statistical_component : ranking function
            A function that has an enspara msm object as input, and
            returns the unique states and statistical rankings per
            state.
        statistical_scaling : scaling object, default = feature_scale(maximize=False)
            An object used for scaling the statistical component values.
        alpha : float, default = 1
            The weighting of the statistical component to FAST.
            i.e. r_i = directed + alpha * undirected
        alpha_percent : bool, default=False
            Optionally treat the alpha value as a percent.
            i.e. r_i = (1 - alpha) * directed + alpha * undirected
        """
        self.state_rankings = state_rankings
        self.directed_scaling = directed_scaling
        self.statistical_component = statistical_component
        self.statistical_scaling = statistical_scaling
        self.alpha = alpha
        self.alpha_percent = alpha_percent
        if self.alpha_percent and ((self.alpha < 0) or (self.alpha > 1)):
            raise
        base_ranking.__init__(self, maximize_ranking=maximize_ranking)

    def rank(self, msm, unique_states=None):
        # determine unique states
        if unique_states is None:
            unique_states = get_unique_states(msm)
        # get statistical component
        statistical_ranking = self.statistical_component.rank(msm)
        # scale the directed weights
        directed_weights = self.directed_scaling.scale(
            self.state_rankings[unique_states])
        # scale the statistical weights
        statistical_weights = self.statistical_scaling.scale(
            statistical_ranking)
        # determine rankings
        if self.alpha_percent:
            total_rankings = (1-self.alpha)*directed_weights + \
                self.alpha*statistical_weights
        else:
            total_rankings = directed_weights + self.alpha*statistical_weights
        return total_rankings        

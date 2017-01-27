import numpy as np
import itertools

def combinatoric_correction(probs):
    """Corrects for the multiplicity of alphas, which are the ranking
       values of landscapes"""
    if len(probs) > 0:
        nums = np.arange(len(probs))
        correction = np.prod(1-probs)
        for num_combs in range(1,len(nums)):
            combins = [
                [i for i in combins] \
                for combins in list(itertools.combinations(nums, num_combs))]
            for iis_positive in combins:
                iis_neg = np.setdiff1d(nums, iis_positive)
                correction += np.prod(probs[iis_positive])*\
                    np.prod(1-probs[iis_neg])/float(num_combs+1)
        correction += np.prod(probs)/(len(nums)+1)
    else:
        correction = 1.0
    return correction

def _get_state_sel_probs(disc_probs, alphas):
    """Helper function for calc_state_sel_probs. Calculates the probabilities
       of selecting alpha for a single slice of probabilities."""
    P_alphas = np.array([])
    for num in range(len(disc_probs)):
        iis_greater = np.where(alphas>alphas[num])
        iis_equal = np.setdiff1d(np.where(alphas==alphas[num])[0],num)
        correction = combinatoric_correction(disc_probs[iis_equal])
        P_alphas = np.append(
            P_alphas, 
            disc_probs[num]*np.prod(1-disc_probs[iis_greater])*correction)
    return P_alphas

def calc_state_sel_probs(disc_probs, alphas):
    """given the discovery probabilities and alphas values, returns the
       probability of selecting a particular state to resimulate from."""
    state_sel_probs = []
    if len(disc_probs.shape) > 1:
        for num in range(len(disc_probs)):
            state_sel_probs.append(
                _get_state_sel_probs(disc_probs[num],alphas))
    else:
        state_sel_probs.append(_get_state_sel_probs(disc_probs,alphas))
    return np.array(state_sel_probs)

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

def _calc_discover_probs(T, steps=1, clones=1, self_known=True):
    """The probability of discovering state j after S steps from state i
       with M trajectories."""
    discover_probs = _calc_discover_probs_stepping(
        T, steps=steps, self_known=self_known)
    discover_probs = 1-((1-discover_probs)**clones)
    return discover_probs

def _get_corrected_row(P_alphas, discover_probs, row):
    P_alphas_conditional = np.array(P_alphas, copy=True)
    P_alphas_conditional[:,row] = 0
    P_alphas_conditional_normed = P_alphas_conditional.sum(axis=1)
    P_alphas_conditional_normed[np.where(P_alphas_conditional_normed<=0)] = 1
    P_alphas_conditional /= P_alphas_conditional_normed[:,None]
    corrected_row = np.matmul(P_alphas_conditional, discover_probs)
    return corrected_row[:,row]

def calc_discover_probs(
        T, runs=0, steps=1, clones=1, alphas=None, self_known=True):
    """Calculates the probability of discovering state j if sampling from
       state i. This is a general algorithm that works for various simulations,
       of multiple steps, with reseeding simulations per round based on the
       alpha rankings"""
    if alphas is None:
        alphas = np.arange(len(T))
    else:
        alphas = np.array(alphas)
    base_probs = _calc_discover_probs(
        T, steps=steps, clones=clones, self_known=self_known)
    discover_probs = np.array(base_probs, copy=True)
    for run in range(runs):
        P_alphas = calc_state_sel_probs(discover_probs, alphas)
        discover_probs = (1 - ((1 - discover_probs) * np.matmul(P_alphas, (1-base_probs))))
    return discover_probs

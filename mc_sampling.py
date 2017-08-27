import itertools
import numpy as np
from . import rankings
from enspara.msm import builders, MSM, synthetic_data
from multiprocessing import Pool


def _run_sampling(adaptive_sampling_obj):
    assignments = adaptive_sampling_obj[0].run()
    return assignments

def adaptive_sampling(
        T, initial_state=0, n_runs=1, n_clones=1, n_steps=1,
        msm_obj=None, ranking_obj=None, n_reps=1, n_procs=1):
    """Get synthetic adaptive sampling run from an MSM

    Parameters
    ----------
    T : array, shape=(n_states, n_states)
        The transition probability matrix from which to sample.
    initial_state : int, default=0
        The initial state from which to start simulations.
    n_runs : int, default=1
        The number of rounds of adaptive sampling.
    n_clones : int, default=1
        The number of clones per run of adaptive sampling.
    n_steps : int, default=1
        The number of steps per clone (each trajectory).
    msm_obj : enspara.msm.MSM object
        An enspara MSM object. This is used to fit assignments at each
        round of sampling.
    ranking_obj : rankings object
        This is an object with at least two functions: __init__(**args)
        and select_states(msm, n_clones). The output of this object is
        a list of states to simulate.
    n_reps : int, default=1
        The number of repetitions of adaptive sampling to perform.
    n_procs : int, default=1
        The number of processes to use when doing adaptive sampling.
        This parallelizes over the number of reps.

    Returns
    ----------
    assignments : array, shape=(n_reps, n_runs, n_clones, n_steps)
       The assignments files for adaptive sampling runs.
    """
    if msm_obj is None:
        MSM(lag_time=1, method=builders.normalize, max_n_states=None)
    if ranking_obj is None:
        ranking_obj = rankings.min_counts()
        
    sampling_info = list(
        zip(
            itertools.repeat(
                Adaptive_Sampling(
                    T, initial_state, n_runs, n_clones, n_steps, msm_obj,
                    ranking_obj),
                n_reps)))
    pool = Pool(processes = n_procs)
    assignments = pool.map(_run_sampling, sampling_info)
    pool.terminate()
    return np.array(assignments)


class Adaptive_Sampling:
    def __init__(
            self, T, initial_state, n_runs, n_clones, n_steps, msm_obj,
            ranking_obj):
        self.T = T
        self.initial_state = initial_state
        self.n_runs = n_runs
        self.n_clones = n_clones
        self.n_steps = n_steps
        self.msm_obj = msm_obj
        self.ranking_obj = ranking_obj

    def run(self):
        np.random.seed()
        assignments = []
        initial_assignments = np.array(
            [
                synthetic_data.synthetic_trajectory(
                    self.T, self.initial_state, self.n_steps)
                for i in range(self.n_clones)])
        assignments.append(initial_assignments)
        for run_num in range(1, self.n_runs):
            self.msm_obj.fit(np.concatenate(assignments))
            state_to_simulate = self.ranking_obj.select_states(
                self.msm_obj, self.n_clones)
            new_assignments = np.array(
                [
                    synthetic_data.synthetic_trajectory(
                        self.T, state_to_simulate[i], self.n_steps)
                    for i in range(self.n_clones)])
            assignments.append(new_assignments)
        return assignments

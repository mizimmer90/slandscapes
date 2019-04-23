import itertools
import numpy as np
import scipy
from . import rankings
from enspara.msm import builders, MSM, synthetic_data
from functools import partial
from multiprocessing import Pool

def synthetic_conv(T, start_state, n_steps, end_state=None):
    """Simulate a single trajectory using kinetic Monte Carlo.

    Parameters
    ----------
    T : array, shape=(n_states, n_states)
        A row-normalized transition probability matrix.
    start_state : int
        State to start the trajectory from.
    n_steps : int
        Number of steps in the trajectory. This includes the starting state,
        so n_steps=2 would result in a trajectory consisting of the starting
        state and one additional state.

    Returns
    -------
    traj : array, shape=(n_steps, )
        A 1-D array containing a sequence of state indices (integers).
    """
    np.random.seed()
    if end_state:
        traj = np.array([start_state])
    else:
        traj = -1*np.ones(n_steps, dtype=int)
        traj[0] = start_state

    for i in range(n_steps-1):
        current_state = traj[i]
        if scipy.sparse.isspmatrix(T):
            p = T[current_state, :].toarray()[0]
        else:
            p = T[current_state, :]
        new_state = np.where(scipy.random.multinomial(1, p) == 1)[0][0]
        if end_state:
            traj = np.append(traj,new_state)
            if end_state == new_state:
                break
        else:
            traj[i+1] = new_state

    return traj

def conv_sim(T, start_state, n_steps, n_cores, n_reps, end_state=None):
    p = Pool(processes = n_cores)
    inputs = list(itertools.repeat((T, start_state, n_steps, end_state), n_reps))
    assignments = p.starmap(synthetic_conv, inputs)
    p.close()
    return assignments


def _run_sampling(adaptive_sampling_obj):
    """Helper to adaptive sampling. Helps parallelize sampling runs."""
    assignments = adaptive_sampling_obj[0].run()
    return assignments


def adaptive_sampling(
        T, initial_state=0, n_runs=1, n_clones=1, n_steps=1,
        msm_obj=None, ranking_obj=None, n_reps=1, n_procs=1,
        assignments=None):
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
    assignments : array-like, shape = (n_trajs, n_steps), default=None
        Optionally provide assignments to continue sampling from.

    Returns
    ----------
    assignments : array, shape=(n_reps, n_runs, n_clones, n_steps)
       The assignments files for adaptive sampling runs.
    """
    if msm_obj is None:
        builder_obj = partial(builders.normalize, calculate_eq_probs=False)
        msm_obj = MSM(
            lag_time=1, method=builder_obj, max_n_states=len(T))
    if msm_obj.max_n_states != len(T):
        print(
            "MSM.max_n_states should be equal to the total number of" + \
            "states. Changing value.")
        msm_obj.max_n_states = len(T)
    if ranking_obj is None:
        ranking_obj = rankings.counts()
        
    sampling_info = list(
        zip(
            itertools.repeat(
                Adaptive_Sampling(
                    T, initial_state, n_runs, n_clones, n_steps, msm_obj,
                    ranking_obj, assignments),
                n_reps)))
    pool = Pool(processes = n_procs)
    new_assignments = pool.map(_run_sampling, sampling_info)
    pool.terminate()
    return np.array(new_assignments)


class Adaptive_Sampling:
    """Adaptive sampling object. Runs a single adaptive sampling run.

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
    assignments : array-like, shape = (n_trajs, n_steps), default=None
        Optionally provide assignments to continue sampling from. If
        using previous assignments, number of steps for each trajectory
        must be the same.

    Returns
    ----------
    assignments : array, shape=(n_runs, n_clones, n_steps)
       The assignments files for adaptive sampling runs.
    """

    def __init__(
            self, T, initial_state, n_runs, n_clones, n_steps, msm_obj,
            ranking_obj, assignments=None):
        # Initialize class variables
        self.T = T
        self.initial_state = initial_state
        self.n_runs = n_runs
        self.n_clones = n_clones
        # adds 1 to the number of steps. This is because synthetic
        # trajectories counts number of states, not steps.
        self.n_steps = n_steps + 1
        self.msm_obj = msm_obj
        self.ranking_obj = ranking_obj
        # format initial assignments if present
        if assignments is not None:
            if len(assignments.shape) == 2:
                pass
            elif len(assignments.shape) == 3:
                pass
            elif (len(assignments.shape) == 4) and (assignments.shape[0] == 1):
                assignments = np.concatenate(assignments)
            else:
                raise
        self.starting_assignments = assignments

    def run(self):
        # initialize random seed. This is necessary for getting
        # independent samplings through parallelization.
        np.random.seed()
        # initialize first run
        assignments = []
        if self.starting_assignments is None:
            initial_assignments = np.array(
                [
                    synthetic_data.synthetic_trajectory(
                        self.T, self.initial_state, self.n_steps)
                    for i in range(self.n_clones)])
            assignments.append(initial_assignments)
            # If there are no starting assignments, gets initial
            # assignments from initial state and this counts as a single
            # run of adaptive sampling.
            run_start = 1
        else:
            # checks if previous assignments have multiple runs (purely
            # for the formatting of output)
            if len(self.starting_assignments.shape) == 3:
                for assignment in self.starting_assignments:
                    assignments.append(assignment)
            else:
                assignments.append(self.starting_assignments)
            run_start = 0
        # iterate through each run and append assignments
        for run_num in range(run_start, self.n_runs):
            # fit assignments with msm object
            self.msm_obj.fit(np.concatenate(assignments))
            # rank states based on ranking object
            states_to_simulate = self.ranking_obj.select_states(
                self.msm_obj, self.n_clones)
            new_assignments = np.array(
                [
                    synthetic_data.synthetic_trajectory(
                        self.T, states_to_simulate[i], self.n_steps)
                    for i in range(self.n_clones)])
            assignments.append(new_assignments)
        assignments = np.array(assignments)
        return assignments

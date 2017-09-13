import numpy as np

def _get_reactive_path(assignment, sinks, source=None):
    if source is None:
        source = assignment[0]

    # determine the first frame that the trajectory reaches a sink
    first_sinks_ii = np.array([],dtype=int)

    # search over all sinks
    for sink in sinks:
        first_sinks_ii = np.concatenate(
            [first_sinks_ii, np.where(assignment==sink)[0]])
    if len(first_sinks_ii) > 0:
        first_sinks_ii = np.min(first_sinks_ii)
        # clip trajectory and find first frame that is reactive
        pathway = assignment[:first_sinks_ii]
        first_source_ii = np.where(pathway == source)[0][-1] + 1
        pathway = pathway[first_source_ii:]
    else:
        pathway = []

    return pathway


def reactive_pathways(assignments, sinks, source=None, verbose=False):
    """
    """
    sinks = np.array(sinks).reshape((-1,))
    pathways = np.array(
        [_get_reactive_path(assignment, sinks) for assignment in assignments])

    # trim empty pathways
    pathway_lengths = np.array([len(pathway) for pathway in pathways])
    pathways = pathways[np.where(pathway_lengths >0)[0]]
    if verbose:
        print(
            "%d reactive trajectories out of %d" % \
            (len(pathways), len(assignments)))

    return pathways


def reactive_density(
        assignments, sinks, source=None, all_reactive=False, verbose=False):
    """
    """
    if not all_reactive:
        pathways = reactive_pathways(assignments, sinks, verbose=verbose)
    else:
        pathways = assignments

    state_counts = np.array(
        np.bincount(np.concatenate(pathways)))
    densities = state_counts / np.sum(state_counts)

    return densities


def state_pathway_prob(
        assignments, sinks, source=None, all_reactive=False, verbose=False):
    """
    """
    if not all_reactive:
        pathways = reactive_pathways(assignments, sinks, verbose=verbose)
    else:
        pathways = assignments

    unique_pathways = np.array(
        [np.unique(pathway) for pathway in pathways])
    state_counts = np.array(np.bincount(np.concatenate(unique_pathways)))
    pathway_probs = state_counts / len(unique_pathways)

    return pathway_probs


def _condition_assignments(assignments, end_state):
    """Chops each assignment off at round that state is discovered.
    Also, it flattens the assignments."""
    state_exists_iis = np.unique(np.where(assignments == end_state)[0])
    conditioned_assignments = []
    for ii in state_exists_iis:
        if (len(assignments.shape) == 4) and (assignments.shape[1:3] == (1,1)):
            cut = np.where(assignments[ii]==end_state)[-1][0] + 1
            conditioned_assignments.append(assignments[ii, 0, 0, :cut])
        else:
            cut = np.where(assignments[ii]==end_state)[0][0] + 1
            conditioned_assignments.append(assignments[ii, :cut])
    state_doesnt_exist_iis = np.setdiff1d(
        np.arange(len(assignments)), state_exists_iis)
    for ii in state_doesnt_exist_iis:
        conditioned_assignments.append(assignments[ii])
    flattened_assignments = np.array(
        [ass.flatten() for ass in conditioned_assignments])
    return flattened_assignments


def discover_probabilities(assignments, n_states=None, end_state=None):
    """Returns the probability that a state is discovered from a set
    of trajectories or sampling run (treats each row in assignments as
    an independent sampling run for calculating probabilities)"""
    if n_states is None:
        n_states = np.max(np.flatten(assignments))
    if end_state is not None and assignments.shape:
        assignments = _condition_assignments(assignments, end_state)
    if len(assignments.shape) > 2:
        assignments = np.array([ass.flatten() for ass in assignments])
    observed_states = np.array(
        [
            np.bincount(ass, minlength=n_states) for ass in assignments]) > 0
    discover_probs = np.sum(observed_states, axis=0) / len(assignments)
    return discover_probs
    

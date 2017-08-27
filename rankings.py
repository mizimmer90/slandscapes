import numpy as np


class min_counts:
    def __init__(self):
        pass

    def select_states(self, msm, n_clones):
        counts_per_state = np.array(msm.tcounts_.sum(axis=1))[:, 0]
        unique_states = np.where(counts_per_state > 0)[0]
        iis = np.argsort(counts_per_state[unique_states])
        return unique_states[iis][:n_clones]

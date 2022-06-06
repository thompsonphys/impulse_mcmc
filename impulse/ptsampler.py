import numpy as np
from impulse.random_nums import rng

# TODO: make the following into their own class called PTSwap or something:
# we need to keep up with some bits here and it would be easier with a class

class PTSwap():

    def __init__(self, ndim, ntemps, tmin=1, tmax=None, tstep=None,
                 tinf=False, adaptation_time=1e2, adaptation_lag=1e3,
                 ladder=None):
        self.ndim = ndim
        self.ntemps = ntemps
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        if ladder is None:
            self.ladder = self.temp_ladder()
        else:
            self.ladder = ladder
            self.ntemps = len(ladder)
        if tinf:
            ladder = np.delete(ladder, -1)  # remove last element
            ladder = np.append(ladder, np.inf)  # replace with inf
        self.swap_accept = np.zeros(ntemps - 1)  # swap acceptance between chains
        self.adaptation_time = adaptation_time
        self.adaptation_lag = adaptation_lag
        self.nswaps = 0


    def temp_ladder(self):
        """
        Method to compute temperature ladder. At the moment this uses
        a geometrically spaced temperature ladder with a temperature
        spacing designed to give 25 % temperature swap acceptance rate.
        """
        if self.tstep is None and self.tmax is None:
            self.tstep = 1 + np.sqrt(2 / self.ndim)
        elif self.tstep is None and self.tmax is not None:
            self.tstep = np.exp(np.log(self.tmax / self.tmin) / (self.tmaxntemps - 1))
        ii = np.arange(self.ntemps)
        ladder = self.tmin * self.tstep**ii
        return ladder


    def adapt_ladder(self, sample_num, adaptation_lag=1e3, adaptation_time=1e2):
        """
        Adapt temperatures according to arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>.
        """
        # Modulate temperature adjustments with a hyperbolic decay.
        decay = adaptation_lag / (sample_num + adaptation_lag)  # t0 / (t + t0)
        kappa = decay / adaptation_time  # 1 / nu
        # Construct temperature adjustments.
        accept_ratio = self.compute_accept_ratio()
        dSs = kappa * (accept_ratio[:-1] - accept_ratio[1:])  # delta acceptance ratios for chains
        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(self.ladder[:-1])
        deltaTs *= np.exp(dSs)
        self.ladder[1:-1] = 1 / (np.cumsum(deltaTs) + self.ladder[0])


    def compute_accept_ratio(self):
        return self.swap_accept / self.nswaps


    def __call__(self, chain, lnlike, swap_idx):  # propose swaps!
        self.nswaps += 1
        lnchainswap = (1 / self.ladder[:-1] - 1 / self.ladder[1:]) * (lnlike[-1, :-1] - lnlike[-1, 1:])
        # lnchainswap = np.nan_to_num(lnchainswap)
        nums = np.log(rng.random(size=len(self.ladder) - 1))
        idxs = np.where(lnchainswap > nums)[0] + 1
        # if not idxs.size == 0:
        self.swap_accept[idxs - 1] += 1
        chain[swap_idx, :, [idxs - 1, idxs]] = chain[swap_idx, :, [idxs, idxs - 1]]
        return chain

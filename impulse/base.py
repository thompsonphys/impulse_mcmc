import numpy as np
from numpy.random import SeedSequence, default_rng
from tqdm import tqdm
from loguru import logger
import ray

from impulse.ptsampler import PTSwap
from impulse.proposals import JumpProposals, am, scam, de
from impulse.save_data import SaveData
from impulse.mhsampler import mh_step

class Sampler():
    def __init__(self,
                 ndim,
                 logl,
                 logp,
                 ncores=1,
                 ntemps=1,
                 tmin=1,
                 tmax=None,
                 tstep=None,
                 tinf=False,
                 adapt=False,
                 adapt_t0=100,
                 adapt_nu=10,
                 ladder=None,
                 swap_steps=100,
                 seed=None,
                 buf_size=50_000,
                 mean=None,
                 cov=None,
                 groups=None,
                 loglargs=[],
                 loglkwargs={},
                 logpargs=[],
                 logpkwargs={},
                 cov_update=1000,
                 save_freq=1000,
                 SCAMweight=30,
                 AMweight=15,
                 DEweight=50,
                 thin=1,
                 outdir="./chains"):

        self.ndim = ndim
        self.ncores = ncores
        self.ntemps = ntemps
        self.tmin = tmin
        self.tmax = tmax
        self.tstep = tstep
        self.tinf = tinf
        self.adapt = adapt
        self.adapt_t0 = adapt_t0
        self.adapt_nu = adapt_nu
        self.ladder = ladder
        self.swap_steps = swap_steps
        self.thin = thin

        # initialize ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=ncores)

        # setup random number generator
        stream = SeedSequence(seed)
        seeds = stream.generate_state(ntemps + 1)
        self.rng = [default_rng(s) for s in seeds]

        # setup parallel tempering
        self.ptswap = PTSwap.remote(self.ndim, self.ntemps, self.rng[-1], tmin=self.tmin, tmax=self.tmax, tstep=self.tstep,
                                    tinf=self.tinf, adapt_t0=self.adapt_t0, adapt_nu=self.adapt_nu, swap_steps=self.swap_steps,
                                    ladder=self.ladder)
        self.ladder = ray.get(self.ptswap.get_temp_ladder.remote())

        # setup samplers:
        self.samplers = [PTSampler.remote(self.ndim, logl, logp, chain_num=ii, temperature=self.ladder[ii], buf_size=buf_size, mean=mean,
                                          cov=cov, groups=groups, loglargs=loglargs, loglkwargs=loglkwargs, logpargs=logpargs, logpkwargs=logpkwargs,
                                          cov_update=cov_update, save_freq=save_freq, SCAMweight=SCAMweight, AMweight=AMweight, DEweight=DEweight,
                                          outdir=outdir, rng=self.rng[ii], ptswap=self.ptswap, thin=self.thin) for ii in range(self.ntemps)]

        # initialize chains
        self.x0 = np.zeros((self.ntemps, self.swap_steps, self.ndim))
        self.lnlike0 = np.zeros((self.ntemps, self.swap_steps))
        self.lnprior0 = np.zeros((self.ntemps, self.swap_steps))
        self.accept = np.zeros((self.ntemps, self.swap_steps))
        self.counter = 0  # counter for iterations

    def sample(self, x0, num_samples, ret_chain=False):
        self.x0[:, 0, :] = x0
        if ret_chain:
            full_chain = np.zeros((self.ntemps, num_samples, self.ndim))

        for ii in tqdm(range(num_samples // self.swap_steps)):
            res = ray.get([sampler.sample.remote(self.x0[jj, self.counter % self.swap_steps, :], self.swap_steps, ret_chain=True) for (jj, sampler) in enumerate(self.samplers)])
            for jj in range(self.ntemps):
                self.x0[jj] = res[jj][0]
                self.lnlike0[jj] = res[jj][1]
                self.lnprior0[jj] = res[jj][2]
                self.accept[jj] = res[jj][3]

            # PT swap
            if self.counter > 1 and self.ntemps > 1:
                self.x0[:, -1, :], self.lnlike0[:, -1] = ray.get(self.ptswap.swap.remote(self.x0[:, self.counter % self.swap_steps, :], self.lnlike0[:, -1]))
                if self.adapt:
                    self.ptswap.adapt_ladder.remote()

            if ret_chain:
                full_chain[:, ii * self.swap_steps:(ii + 1) * self.swap_steps, :] = self.x0

            self.counter += 1

        ray.shutdown()
        if ret_chain:
            return full_chain

    def save_state(self):
        pass

    def load_state(self):
        pass
        

@ray.remote(num_cpus=1)
class PTSampler():
    def __init__(self,
                 ndim,
                 logl,
                 logp,
                 chain_num=1,
                 temperature=1,
                 buf_size=50_000,
                 mean=None,
                 cov=None,
                 groups=None,
                 loglargs=[],
                 loglkwargs={},
                 logpargs=[],
                 logpkwargs={},
                 cov_update=1000,
                 save_freq=1000,
                 SCAMweight=30,
                 AMweight=15,
                 DEweight=50,
                 thin=1,
                 outdir="./chains",
                 rng=np.random.default_rng(),
                 ptswap=None):

        # setup loglikelihood and logprior functions
        self.logl = _function_wrapper(logl, loglargs, loglkwargs)
        self.logp = _function_wrapper(logp, logpargs, logpkwargs)

        # PTSwap parameters
        self.chain_num = chain_num  # number of this chain
        self.temp = temperature  # temperature of this chain

        # other important bits
        self.outdir = outdir
        self.ndim = ndim
        self.cov_update = cov_update
        self.save_freq = save_freq
        self.rng = rng
        self.ptswap = ptswap
        self.thin = thin

        # sample counter
        self.counter = 0

        # setup standard jump proposals
        self.mix = JumpProposals(self.ndim, buf_size=buf_size, groups=groups, cov=cov, mean=mean)
        self.mix.add_jump(scam, SCAMweight)
        self.mix.add_jump(am, AMweight)
        self.mix.add_jump(de, DEweight)

        if ptswap is not None:
            self.swap_steps = ray.get(ptswap.get_swap_steps.remote())
        else:
            self.swap_steps = 1e80  # never swap

        # setup save
        self.filename = '/chain_{}.txt'.format(self.chain_num) # temps change (label by chain number)
        self.save = SaveData(outdir=self.outdir, filename=self.filename, thin=thin)

        # setup arrays
        self.chain = np.zeros((self.save_freq, self.ndim))
        self.lnlike_arr = np.zeros(self.save_freq)
        self.lnprob_arr = np.zeros(self.save_freq)
        self.accept_arr = np.zeros(self.save_freq)

    def update(self, x0, lnlike0, lnprob0):
        self.x0 = x0
        self.lnlike0 = lnlike0
        self.lnprob0 = lnprob0

    def sample(self, x0, num_samples, ret_chain=False):
        if ret_chain:
            full_chain = np.zeros((num_samples, self.ndim))
            full_like = np.zeros(num_samples)
            full_prob = np.zeros(num_samples)
            full_accept = np.zeros(num_samples)

        # initial sample
        self.x0 = np.array(x0)  # (ntemps, ndim)
        self.lnlike0 = self.logl(self.x0)
        self.lnprob0 = self.logp(self.x0) + self.lnlike0

        # start sampling!
        for ii in range(num_samples):
            kk = self.counter % self.save_freq
            self.counter += 1

            # metropolis hastings step + update chains
            self.x0, self.lnlike0, self.lnprob0, self.accept = mh_step(self.x0, self.lnlike0, self.lnprob0, self.logl,
                                                                       self.logp, self.mix, self.rng, self.temp)
            self.chain[kk, :] = self.x0
            self.lnlike_arr[kk] = self.lnlike0
            self.lnprob_arr[kk] = self.lnprob0
            self.accept_arr[kk] = self.accept

            # PT swap!
            # if self.counter % self.swap_steps == 0 and self.counter > 1:
            #     self.x0, self.lnlike0 = ray.get(self.ptswap.swap.remote(self.chain[kk, :], self.lnlike_arr[kk]))
            #     self.lnprob0 = self.logp(self.x0) + self.lnlike0

            # update covariance matrix
            if self.counter % self.cov_update == 0 and self.counter > 1:
                self.mix.recursive_update(self.counter, self.chain)

            # save and save state of PTSampler
            if self.counter % self.save_freq == 0 and self.counter > 1:
                self.save(self.chain, self.lnlike_arr, self.lnprob_arr, self.accept_arr)

            if ret_chain:
                full_chain[ii] = self.x0
                full_like[ii] = self.lnlike0
                full_prob[ii] = self.lnprob0
                full_accept[ii] = self.accept

        if ret_chain:
            return full_chain, full_like, full_prob, full_accept


class _function_wrapper(object):

    """
    This is a hack to make the likelihood function pickleable when ``args``
    or ``kwargs`` are also included.
    """

    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.f(x, *self.args, **self.kwargs)

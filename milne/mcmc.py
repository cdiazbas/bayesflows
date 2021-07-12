import pymilne_ as pymilne
import numpy as np
import matplotlib.pyplot as pl
import time
import emcee
import corner

my_seed = 8
np.random.seed(my_seed)

class Inference(object):
    def __init__(self):
        lambda0 = 6301.5080
        JUp = 2.0
        JLow = 2.0
        gUp = 1.5
        gLow = 1.833
        lambdaStart = 6300.9
        lambdaStep = 0.04
        self.nLambda = 30

        lineInfo = np.asarray([lambda0, JUp, JLow, gUp, gLow, lambdaStart, lambdaStep])

        wav = pymilne.init(self.nLambda, lineInfo)

    def gen_obs(self, noise):
        self.noise = noise

        BField = 0.0
        BTheta = 0.0
        BChi = 0.0
        VMac = -1.0
        damping = 0.1
        damping = 0.0
        B0 = 0.35
        B1 = 0.7
        beta = B1 / B0
        VDop = 0.15
        kl = 2.5


        self.x0 = np.array([VMac, VDop, kl, B0, B1])

        modelSingle = np.asarray([BField, BTheta, BChi, VMac, damping, B0, B0*beta, VDop, kl])

        self.obs = pymilne.synth(self.nLambda, modelSingle, 1.0)[0, :]
        
        self.obs_noise = self.obs + noise * np.random.randn(self.nLambda)        

    def synth(self, x):
        BField = 0.0
        BTheta = 0.0
        BChi = 0.0
        # VMac = -1.0
        damping = 0.0
        # B0 = 0.8
        # B1 = 0.2        
        # VDop = 0.12
        # kl = 3.7

        VMac = x[0]
        VDop = x[1]
        kl = x[2]
        B0 = x[3]
        B1 = x[4]

        modelSingle = np.asarray([BField, BTheta, BChi, VMac, damping, B0, B1, VDop, kl])

        syn = pymilne.synth(self.nLambda, modelSingle, 1.0)[0, :]

        return syn

    def log_prob(self, x):		
        BField = 0.0
        BTheta = 0.0
        BChi = 0.0
        # VMac = -1.0
        damping = 0.0
        # B0 = 0.8
        # B1 = 0.2
        # VDop = 0.12
        # kl = 3.7

        VMac = x[0]
        VDop = x[1]
        kl = x[2]
        B0 = x[3]
        B1 = x[4]

        if (VMac < -3.0 or VMac > 3.0):
            return -np.inf

        if (damping < 0.0 or damping > 0.2):
            return -np.inf

        if (B0 <= 0.0 or B0 > 1.0):
            return -np.inf

        if (B1 <= 0.0 or B1 > 1.0):
            return -np.inf

        if (VDop <= 0.05 or VDop > 0.2):
            return -np.inf

        if (kl <= 0.0 or kl > 5.0):
            return -np.inf		

        syn = self.synth(x)

        chi2 = np.sum( (syn - self.obs_noise)**2 / self.noise**2 )

        return -0.5 * chi2

    def sample(self):
        ndim = 5
        nwalkers = 1000
        p0 = self.x0[None, :] + 1e-2*np.random.randn(nwalkers, ndim)
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_prob)

        self.sampler.run_mcmc(p0, 1000, progress=True)

        self.samples = self.sampler.get_chain(discard=100, flat=True)

        np.savez('samples.npz', samples=self.samples, obs=self.obs, obs_noise=self.obs_noise, x0=self.x0)

        corner.corner(self.samples, truths=self.x0, show_titles=True, plot_datapoints=False, fill_contours=True, bins=50, smooth=2.0)
        pl.savefig('mcmc_corner.pdf')

        fig, ax = pl.subplots()
        ax.plot(self.obs_noise,'k')

        for i in range(100):
            stokes = self.synth(self.samples[i, :])
            ax.plot(stokes, alpha=0.1, color='C0')

        # pl.show()
        pl.savefig('mcmc_plot.pdf')


if (__name__ == '__main__'):
    tmp = Inference()
    tmp.gen_obs(8e-3)
    tmp.sample()
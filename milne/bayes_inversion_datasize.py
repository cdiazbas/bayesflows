# -*- coding: utf-8 -*-
__author__ = 'carlos.diaz'
# Density estimation in solar inversions - MILNE EDDINGTON example

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import time
import os
import nde_utils
import nde_nflow
from tqdm import tqdm
import pymilne_ as pymilne
import corner
from ipdb import set_trace as stop

# nde_utils.fix_seed(0)

class bayes_inversion(object):

    # =========================================================================
    def __init__(self, directory = 'bayes_inversion_output_curve/', device = 'cpu'):

        # Configuration
        self.args = nde_utils.dotdict()
        self.args.kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}
        self.args.directory = directory
        self.device = device
        if not os.path.exists(self.args.directory): os.makedirs(self.args.directory)


    # =========================================================================
    def create_database(self, batch_size = 100, n = 100, noise=8e-3):

        self.n_training = n

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

        BField = 0.0
        BTheta = 0.0
        BChi = 0.0
        damping = 0.0

        VMac = np.random.uniform(low=-3.0, high=3.0, size=n)
        VDop = np.random.uniform(low=0.05, high=0.2, size=n)
        kl = np.random.uniform(low=0.0, high=5.0, size=n)        
        B0 = np.random.uniform(low=0.0, high=1.0, size=n)
        B1 = np.random.uniform(low=0.0, high=1.0, size=n)

        self.n_pars = 5
        self.pars = np.zeros((n, self.n_pars))
        self.pars[:, 0] = VMac
        self.pars[:, 1] = VDop
        self.pars[:, 2] = kl
        self.pars[:, 3] = B0
        self.pars[:, 4] = B1
        
        for i in tqdm(range(n)):	
            modelSingle = np.asarray([BField, BTheta, BChi, VMac[i], damping, B0[i], B1[i], VDop[i], kl[i]])
            syn = pymilne.synth(self.nLambda, modelSingle, 1.0)[0, :]
            if (i == 0):
                self.stokes = np.zeros((n, len(syn)))
            self.stokes[i, :] = syn

        self.n_lambda = self.stokes.shape[1]

        lines = self.stokes.astype(np.float32)
        values = self.pars.astype(np.float32)

        split = 0.9
        train_split = int(lines.shape[0]*split)
        wholedataset = np.arange(lines.shape[0])
        np.random.shuffle(wholedataset)

        self.args.batch_size = batch_size
        self.train_loader = nde_utils.basicLoader(lines[wholedataset[:train_split],:], values[wholedataset[:train_split],:], noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)

        self.vali_loader = nde_utils.basicLoader(lines[wholedataset[train_split:],:], values[wholedataset[train_split:],:], noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)

        print("[INFO] len(pars):", self.n_pars)
        print("[INFO] len(waves):", self.n_lambda)
        print('[INFO] Datasize obsdata: ',lines.shape)
        print('[INFO] Datasize params:  ',values.shape)
        print('[INFO] Train/valid split: ',train_split,int(lines.shape[0]*(1.0-split)))



    # =========================================================================
    def train_network(self, num_epochs = 2000, learning_rate = 1e-4, log_interval = 1, continueTraining=True, name_posterior= 'posterior',num_flows=5,num_blocks=1,mhidden_features=32,nbins=8):
        
        self.args.y_size = self.n_pars
        self.args.x_size = self.n_lambda
        self.model = nde_nflow.NFLOW(self.args.y_size, self.args.x_size,num_flows=num_flows, mhidden_features=mhidden_features, num_blocks=num_blocks, 
                    train_loader=self.train_loader, embedding_net=None, transformtype="rq-coupling",nbins=nbins)
        nde_utils.get_params(self.model)

        self.args.learning_rate = learning_rate
        self.args.num_epochs = num_epochs
        self.args.log_interval = log_interval
        self.args.name_posterior = name_posterior
        print('[INFO] name_posterior: ',name_posterior)

        if continueTraining: self.model = torch.load(self.args.directory+name_posterior+'_best.pth'); print('Loading previous weigths...')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        train_loss_avg = []
        vali_loss_avg = []
        time0 = time.time()
        
        self.valimin = 1e3
        self.waitloop = 0
        self.maxiloop = 10000
        from tqdm import trange
        t = trange(num_epochs, desc='', leave=True)
        for epoch in t:
            self.model.train()
            avgloss = 0
            for batch_idx, (params, data) in enumerate(tqdm(self.train_loader, desc='', leave=False)):
                data = data.to(self.device)
                params = params.to(self.device)
                optimizer.zero_grad()
                loss  = self.model.forward(params, data)
                loss.backward()
                optimizer.step()
                avgloss += loss.item()
                
            avgloss /= (batch_idx+1)
            train_loss_avg.append(avgloss)


            self.model.eval()
            avgloss2 = 0
            for batch_idx, (params, data) in enumerate(self.vali_loader):
                data = data.to(self.device)
                params = params.to(self.device)
                loss  = self.model.forward(params, data)
                avgloss2 += loss.item()
                
            avgloss2 /= (batch_idx+1)
            vali_loss_avg.append(avgloss2)

            argminiv = np.argmin(vali_loss_avg)
            miniv = np.mean(vali_loss_avg[argminiv-1:argminiv+1+1])

            fig = plt.figure(); plt.plot(train_loss_avg); plt.plot(vali_loss_avg)
            plt.axhline(np.mean(train_loss_avg[-10:]),color='C0',ls='--')
            plt.axhline(np.mean(train_loss_avg[-10:]),color='k',ls='--',alpha=0.5)
            plt.axvline(argminiv,color='k',ls='--',alpha=0.5)
            plt.axhline(miniv,color='C1',ls='--')
            plt.axhline(miniv,color='k',ls='--',alpha=0.5)
            plt.title('loss_final: {0:2.2f} / {1:2.2f}'.format( np.mean(train_loss_avg[-10:]), miniv ))
            plt.xlabel('Epochs'); plt.ylabel('Loss')
            plt.savefig(self.args.directory+self.args.name_posterior+'_train_loss_avg.pdf'); plt.close(fig)

            if avgloss2 < self.valimin:
                torch.save(self.model, self.args.directory+self.args.name_posterior+'_best.pth')
                self.valimin = np.copy(avgloss2)
                self.waitloop = 0
            else:
                self.waitloop += 1

            if self.waitloop > self.maxiloop:
                import sys
                print('[INFO] Done')
                sys.exit()

            t.set_postfix({'loss': '{:.2f}'.format(avgloss)})
            t.refresh()



    # =========================================================================
    def test_plots(self, testindex=0,nsamples = 1000):
        tmp = np.load('samples.npz')
        self.samples = tmp['samples']
        self.obs = tmp['obs']
        self.obs_noise = tmp['obs_noise']
        self.x0 = tmp['x0']
        self.n_lambda = len(self.obs)
        # self.model = torch.load(directory+name_posterior+'_best.pth')
        profile = torch.tensor(self.obs[None, :].astype('float32'))
        print('Running samples like the mcmc:',self.samples.shape[0])
        nsamples = int(self.samples.shape[0])
        output = self.model.obtain_samples(profile,nsamples).data.cpu().numpy()

        color1 = 'C5' #C5
        color2 = 'C1' #C1
        labels = ['v [km/s]', r'$\Delta v$', r'$k_l$', r'$S_0$', r'$S_1$']
        weights = np.ones(self.samples.shape[0])*output.shape[0]/self.samples.shape[0]
        # https://corner.readthedocs.io/en/latest/pages/sigmas.html
        fig = corner.corner(self.samples, labels=labels, truths=self.x0, color=color1, show_titles=False, plot_datapoints=False, 
            fill_contours=True, bins=50, smooth=2.0, weights=weights,levels=(0.39,0.86*0.92,),alpha=0.1,contourf_kwargs={"antialiased":True})

        _ = corner.corner(output, labels=labels, truths=self.x0, fig=fig, color=color2, show_titles=False, plot_datapoints=False, 
            fill_contours=True,no_fill_contours=True, bins=50, smooth=2.0,truth_color="slategrey",levels=(0.39,0.86*0.92,),contourf_kwargs={"antialiased":True})

        plt.savefig('corner_flow.pdf')
        plt.savefig(self.args.directory+self.args.name_posterior+'_im_plot_nn.pdf')
        plt.close(fig)




    # =========================================================================
    def test_profile(self, nsamples = 1000,name_posterior='posterior',directory='.', testindex_array=None, nprofiles=1000, ncheck = 100):

        # Create database:
        self.create_database(batch_size = 100, n = nprofiles, noise=8e-3)

        #Init:
        lambda0 = 6301.5080
        JUp = 2.0
        JLow = 2.0
        gUp = 1.5
        gLow = 1.833
        lambdaStart = 6300.9
        lambdaStep = 0.04
        lineInfo = np.asarray([lambda0, JUp, JLow, gUp, gLow, lambdaStart, lambdaStep])
        self.wav = pymilne.init(self.n_lambda, lineInfo)

        BField = 0.0
        BTheta = 0.0
        BChi = 0.0
        damping = 0.0
        mu = 1.0

        # Fix train/valid split
        nprofiles = self.train_loader.dataset.observations.shape[0]

        # globalsigma = []
        globalmaxprob = []
        globaldistri = []
        aa = np.array([],dtype=np.float64)
        bb = np.array([],dtype=np.float64)

        labelstr = [3e2, 1e3, 1e4, 1e5, 1e6]
        datastr = ['_d1e25','_d1e3B','_d1e4','_d1e5','_d1e6B']
        for datastri in datastr:
            maxprob = []
            sigma = []
            for ipix in range(nprofiles):
                profile0 = self.train_loader.dataset.observations[ipix:ipix+1,:]
                profile1 = profile0*1.0 + np.random.normal(0,8e-3,size=profile0.shape)
                profile = torch.tensor(profile1.astype('float32'))
                self.model = torch.load(directory+name_posterior+datastri+'_best.pth')
                time0 = time.time()
                output, logprob = self.model.sample_and_log_prob(profile,nsamples)
                print(f'Samples per second in {name_posterior+datastri}:',int(nsamples/(time.time()-time0)))
                argmaxlogprob = np.argmax(logprob)

                # Synthetizing new profiles:
                for i in range(ncheck):
                    VMac = output[i, 0]
                    VDop = output[i, 1]
                    kl = output[i, 2]
                    B0 = output[i, 3]                        
                    B1 = output[i, 4]
                    
                    modelSingle = np.asarray([BField, BTheta, BChi, VMac, damping, B0, B1, VDop, kl])
                    syn_flow = pymilne.synth(self.n_lambda, modelSingle, mu)[0, :]

                    if (i == 0):
                        stokes_flow = np.zeros((ncheck, len(syn_flow)))                
                    stokes_flow[i, :] = syn_flow

                # One more
                VMac = output[argmaxlogprob, 0]
                VDop = output[argmaxlogprob, 1]
                kl = output[argmaxlogprob, 2]
                B0 = output[argmaxlogprob, 3]                        
                B1 = output[argmaxlogprob, 4]
                
                modelSingle = np.asarray([BField, BTheta, BChi, VMac, damping, B0, B1, VDop, kl])
                syn_flow = pymilne.synth(self.n_lambda, modelSingle, mu)[0, :]

                bb = np.append(bb,syn_flow-profile1)
                aa = np.append(aa,(stokes_flow-profile1).flatten() )
            
            globalmaxprob.append(np.std(bb))
            globaldistri.append(np.std(aa))
            aa = np.array([],dtype=np.float64)
            bb = np.array([],dtype=np.float64)



        plt.clf()
        plt.figure(figsize=(5,4))
        plt.plot(labelstr,globaldistri,'.-',label='distribution')
        plt.plot(labelstr,globalmaxprob,'.-',label='maxlogprob')
        plt.locator_params(axis='y', nbins=4)
        plt.xscale('log')
        plt.xlabel('Dataset size - N')
        plt.ylabel(r'Average error - $\sigma$')
        plt.axhline(8e-3,color='k',ls='--')
        plt.minorticks_on()
        plt.legend()
        plt.ylim(7e-3)
        plt.savefig('plot_datasize.pdf')



if __name__ == "__main__":

    myflow = bayes_inversion()

    myflow.create_database(batch_size=100, n=1000000, noise=8e-3)
    # myflow.train_network(num_epochs=20000, continueTraining=False, learning_rate=1e-4, name_posterior='posterior_5_32_5_b8', num_flows=5, mhidden_features=32, num_blocks=5, nbins=8)

    # myflow.test_profile(name_posterior= 'posterior_5_32_5_b8',directory = 'bayes_inversion_output_final/')




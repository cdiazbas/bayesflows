# -*- coding: utf-8 -*-
__author__ = 'carlos.diaz'
# NN to mimic the forward process (from physical params to Stokes)

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import time
import os
import nde_utils
import nde_cvae
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False

def swish(x):
    return x * F.sigmoid(x)

# from ipdb import set_trace as stop

class bayes_inversion(object):

    # =========================================================================
    def __init__(self, directory = 'bayes_inversion_output_final/', device = 'cpu'):

        # Configuration
        self.args = nde_utils.dotdict()
        self.args.kwargs = {'num_workers': 1, 'pin_memory': True} if device=="cuda" else {}
        self.args.directory = directory
        self.device = device
        if not os.path.exists(self.args.directory): os.makedirs(self.args.directory)



    # =========================================================================
    def create_database(self, batch_size = 100, tauvalues = 15, spectral_range=0, noise=5e-4):

        import sparsetools as sp
        print('[INFO] Using spectral range '+str(spectral_range))
        print('[INFO] Reading database')

        mdir = '../gaussian_model/'
        lines = np.concatenate([np.load(mdir+'trainfixg_lines.npy'),np.load(mdir+'trainfixe_lines.npy'),
        np.load(mdir+'test_lines.npy')])
        values =  np.concatenate([np.load(mdir+'trainfixg_values.npy'),np.load(mdir+'trainfixe_values.npy'),
        np.load(mdir+'test_values.npy')])


        self.waves_info = np.load(mdir+'train_waves_info.npy')
        self.waves = np.load(mdir+'train_waves.npy')
        self.lenwave = len(self.waves)
        self.ltau = np.load(mdir+'train_ltau.npy')
        self.mltau = np.load(mdir+'train_mltau.npy')
        self.lentau = len(self.mltau)
        self.spectral_range = spectral_range
        self.spectral_idx = np.load(mdir+'train_spectral_idx.npy')


        split = 0.9
        train_split = int(lines.shape[0]*split)
        wholedataset = np.arange(lines.shape[0])
        np.random.shuffle(wholedataset)

        self.args.batch_size = batch_size
        # self.train_loader = nde_utils.basicLoader(lines[wholedataset[:train_split],:], values[wholedataset[:train_split],:], noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)
        self.train_loader = nde_utils.basicLoader(lines[:,:], values[:,:], noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)

        self.vali_loader = nde_utils.basicLoader(lines[wholedataset[train_split:],:], values[wholedataset[train_split:],:], noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)

        print("[INFO] len(ltau):", self.lentau)
        print("[INFO] len(waves):", self.lenwave)
        print('[INFO] Datasize obsdata: ',lines.shape)
        print('[INFO] Datasize params:  ',values.shape)
        print('[INFO] Train/valid split: ',train_split,int(lines.shape[0]*(1.0-split)))

        #vali cube:
        print('[INFO] Reading test database')

        mdir = '../gaussian_model/'
        lines = np.load(mdir+'test_lines.npy')
        values = np.load(mdir+'test_values.npy')
        self.test_loader = nde_utils.basicLoader(lines, values, noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)


    # =========================================================================
    def train_network(self, num_epochs = 2000, learning_rate = 1e-6, log_interval = 1, continueTraining=True, name_posterior= 'posterior',num_blocks=1,mhidden_features=32,mse=False,use_batch_norm=False,zero_initialization=True,activation=F.elu,typemodel=1,train_worst=False):
        

        name_posterior = name_posterior+'_sp'+str(self.spectral_range)
        print('[INFO] Network name: ',name_posterior)
        self.args.y_size = self.lentau*3
        self.args.x_size = self.lenwave
        if typemodel == 1: # Gaussian ResNet
            self.model = nde_cvae.ResidualNet(self.args.y_size,self.args.x_size, hidden_features=mhidden_features, num_blocks=num_blocks, 
                    train_loader=self.train_loader,activation=activation,use_batch_norm=use_batch_norm,zero_initialization=zero_initialization)
        elif typemodel == 2: # Simple ResNet
            self.model = nde_cvae.ResidualNet2(self.args.y_size,self.args.x_size, hidden_features=mhidden_features, num_blocks=num_blocks, 
                    train_loader=self.train_loader,activation=activation,use_batch_norm=use_batch_norm,zero_initialization=zero_initialization)            
        elif typemodel == 3: # Fourier ResNet
            self.model = nde_cvae.ResidualNet4(self.args.y_size,self.args.x_size, hidden_features=mhidden_features, num_blocks=num_blocks, 
                    train_loader=self.train_loader,activation=activation,use_batch_norm=use_batch_norm,zero_initialization=zero_initialization)            
        
        nde_utils.get_params(self.model)

        self.args.learning_rate = learning_rate
        self.args.num_epochs = num_epochs
        self.args.log_interval = log_interval
        self.args.name_posterior = name_posterior

        if continueTraining: self.model = torch.load(self.args.directory+name_posterior+'_best.pth'); print('Loading previous weigths...')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        train_loss_avg = []
        vali_loss_avg = []
        time0 = time.time()
        
        if train_worst is False:
            from tqdm import trange
            t = trange(num_epochs, desc='', leave=True)
            for epoch in t:
                self.model.train()
                avgloss = 0
                for batch_idx, (params, data) in enumerate(tqdm(self.train_loader, desc='', leave=False)):
                    data = data.to(self.device)
                    params = params.to(self.device)
                    optimizer.zero_grad()
                    if mse is False:
                        loss  = self.model.forward(data,params)
                    else:
                        loss  = self.model.forward_mse(data,params)
                    loss.backward()
                    optimizer.step()
                    avgloss += loss.item()
                    
                avgloss /= (batch_idx+1)
                torch.save(self.model, self.args.directory+self.args.name_posterior+'_best.pth')
                train_loss_avg.append(avgloss)


                self.model.eval()
                avgloss2 = 0
                for batch_idx, (params, data) in enumerate(self.vali_loader):
                    data = data.to(self.device)
                    params = params.to(self.device)
                    if mse is False:
                        loss  = self.model.forward(data,params)
                    else:
                        loss  = self.model.forward_mse(data,params)
                    avgloss2 += loss.item()
                    
                avgloss2 /= (batch_idx+1)
                vali_loss_avg.append(avgloss2)

                fig = plt.figure(); plt.plot(train_loss_avg); plt.plot(vali_loss_avg)
                plt.axhline(np.mean(train_loss_avg[-10:]),color='C0',ls='--')
                plt.axhline(np.mean(train_loss_avg[-10:]),color='k',ls='--',alpha=0.5)
                plt.axhline(np.mean(vali_loss_avg[-10:]),color='C1',ls='--')
                plt.axhline(np.mean(vali_loss_avg[-10:]),color='k',ls='--',alpha=0.5)
                plt.title('loss_final: {0:2.2e} / {1:2.2e}'.format( np.mean(train_loss_avg[-10:]), np.mean(vali_loss_avg[-10:]) ))
                plt.xlabel('Epochs'); plt.ylabel('Loss')
                if mse is True: plt.yscale('log')
                plt.savefig(self.args.directory+self.args.name_posterior+'_train_loss_avg.pdf')
                plt.close(fig)
                
                self.test_plots(11387, self.args.name_posterior)
                self.test_plots(8178, self.args.name_posterior)

                t.set_postfix({'loss': '{:.2e}'.format(avgloss)})
                t.refresh()

        else:
            
            from tqdm import trange
            t = trange(num_epochs, desc='', leave=True)
            for epoch in t:
            
                # One more round for those worst points.
                self.model.eval()
                maxloss = []
                params = self.train_loader.dataset.modelparameters[:,:]
                mdata = self.train_loader.dataset.observations[:,:]
                output  = self.model.evaluate(params)
                maxloss = np.max(np.abs(mdata-output),axis=1)        
                ll1, ll2 = nde_utils.sorting(maxloss,np.arange(len(maxloss)))
                windex = ll2[::-1]
                nworst = 10000
                
                self.model.train()
                avgloss = 0
                self.worst_loader = nde_utils.basicLoader(self.train_loader.dataset.observations[windex[:nworst],:], self.train_loader.dataset.modelparameters[windex[:nworst],:],noise=0.0, batch_size=100, shuffle=True)

                for batch_idx, (params, data) in enumerate(tqdm(self.worst_loader, desc='', leave=False)):
                    data = data.to(self.device)
                    params = params.to(self.device)
                    optimizer.zero_grad()
                    if mse is False:
                        loss  = self.model.forward(data,params)
                    else:
                        loss  = self.model.forward_mse(data,params)
                    loss.backward()
                    optimizer.step()
                    avgloss += loss.item()
                    
                avgloss /= (batch_idx+1)
                torch.save(self.model, self.args.directory+self.args.name_posterior+'_best.pth')
                train_loss_avg.append(avgloss)

                fig = plt.figure(); plt.plot(train_loss_avg); plt.plot(vali_loss_avg)
                plt.axhline(np.mean(train_loss_avg[-10:]),color='C0',ls='--')
                plt.axhline(np.mean(train_loss_avg[-10:]),color='k',ls='--',alpha=0.5)
                # plt.axhline(np.mean(vali_loss_avg[-10:]),color='C1',ls='--')
                # plt.axhline(np.mean(vali_loss_avg[-10:]),color='k',ls='--',alpha=0.5)
                # plt.title('loss_final: {0:2.2e} / {1:2.2e}'.format( np.mean(train_loss_avg[-10:]), np.mean(vali_loss_avg[-10:]) ))
                plt.xlabel('Epochs'); plt.ylabel('Loss')
                if mse is True: plt.yscale('log')
                plt.savefig(self.args.directory+self.args.name_posterior+'_train_loss_avg.pdf')
                plt.close(fig)
                
                self.test_plots(11387, self.args.name_posterior)
                self.test_plots(8178, self.args.name_posterior)

                t.set_postfix({'loss': '{:.2e}'.format(avgloss)})
                t.refresh()




    # =========================================================================
    def test_plots(self, testindex, name_posterior ,nsamples = 1000):
        inc = 0.8
        fig3 = plt.figure(figsize=(8*inc,14*inc))

        colorsp = ['C0','C4','k']
        colorsp = ['C1','C5','k']
        lsi = ['-','--','.-']
        cc = 0
        
        testvalue = self.test_loader.dataset.modelparameters[testindex,:]
        testobs = self.test_loader.dataset.observations[testindex,:]
        samples_histo = self.model.obtain_samples(testvalue,nsamples)[:,0,:]
        xx = np.arange(len(testobs))

        plt.subplot(411)
        plt.ylim(np.min(testobs)*0.5,np.max(testobs)*1.5)
        plt.plot(xx,testobs, "k", marker='s', markersize=2, label="truth", ls='none')
        plt.fill_between(xx,np.percentile(samples_histo, 16, axis=0),np.percentile(samples_histo, 84, axis=0),alpha=0.4,color=colorsp[cc])
        plt.plot(xx,np.percentile(samples_histo, 50, axis=0),lsi[cc],color=colorsp[cc])
        plt.title('STD: {0:2.2e}, MAX: {1:2.2e}, STDNET: {2:2.2e} '.format(np.std(np.percentile(samples_histo, 50, axis=0)-testobs),
        np.max(np.abs(np.percentile(samples_histo, 50, axis=0)-testobs)),
        np.mean(np.percentile(samples_histo, 84, axis=0)-np.percentile(samples_histo, 16, axis=0)),
        ))
        cc += 1

        plt.savefig(self.args.directory+name_posterior+'_'+str(testindex)+'_paper_im_plot_nn.pdf',bbox_inches='tight')
        plt.close(fig3)




    # =========================================================================
    def test_plots_paper(self, nsamples = 10000, name_posterior = 'posterior',tauvalues = 9,spirange=[0,1],testindex=11387,gotostic = False):

        inc = 0.8
        fig3 = plt.figure(figsize=(8*inc,14*inc))

        colorsp = ['C0','C4','k']
        colorsp = ['C1','C5','k']
        lsi = ['-','--','.-']
        cc = 0

        for spi in spirange:
            self.create_database(tauvalues = tauvalues, spectral_range=spi)
            self.args.y_size = self.lentau*3
            self.args.x_size = self.lenwave
            self.model = torch.load(self.args.directory+name_posterior+'_sp'+str(self.spectral_range)+'_best.pth')
            nde_utils.get_params(self.model)

            self.model.eval()
            maxloss = []; aveloss = []
            
            for ii in tqdm(range(self.test_loader.dataset.modelparameters.shape[0])):
                params = self.test_loader.dataset.modelparameters[ii,:]
                mdata = self.test_loader.dataset.observations[ii,:]
            # for ii in tqdm(range(self.train_loader.dataset.modelparameters.shape[0])):
                # params = self.train_loader.dataset.modelparameters[ii,:]
                # mdata = self.train_loader.dataset.observations[ii,:]
                        
                output  = self.model.evaluate(params)
                maxi = np.max(np.abs(mdata-output))        
                ave = np.mean(np.abs(mdata-output))        #.detach().numpy()
                maxloss.append(maxi)
                aveloss.append(ave)
            maxloss = np.array(maxloss)
            aveloss = np.array(aveloss)
            print('[INFO] Max discrepancy: ',name_posterior,np.max(maxloss))
            print('[INFO] Max ave discrepancy: ',name_posterior,np.mean(maxloss))
            print('[INFO] Average discrepancy: ',name_posterior,np.mean(aveloss))

            testindex = np.argmax(maxloss)

            mltau = self.mltau
            waves = self.waves
            testvalue = self.test_loader.dataset.modelparameters[testindex,:]
            testobs = self.test_loader.dataset.observations[testindex,:]
            samples_histo = self.model.obtain_samples(testvalue,nsamples)[:,0,:]
            xx = np.arange(len(testobs))

            plt.subplot(411)
            plt.ylim(np.min(testobs)*0.5,np.max(testobs)*1.5)
            plt.plot(xx,testobs, "k", marker='s', markersize=2, label="truth", ls='none')
            plt.fill_between(xx,np.percentile(samples_histo, 16, axis=0),np.percentile(samples_histo, 84, axis=0),alpha=0.4,color=colorsp[cc])
            plt.plot(xx,np.percentile(samples_histo, 50, axis=0),lsi[cc],color=colorsp[cc])
            plt.title('STD: {0:2.2e}, MAX: {1:2.2e}, STDNET: {2:2.2e} '.format(np.std(np.percentile(samples_histo, 50, axis=0)-testobs),
            np.max(np.abs(np.percentile(samples_histo, 50, axis=0)-testobs)),
            np.mean(np.percentile(samples_histo, 84, axis=0)-np.percentile(samples_histo, 16, axis=0)),
            ))
            cc += 1

        print(self.args.directory+name_posterior+'_'+str(testindex)+'_'+''.join(map(str, spirange))+'_paper_im_plot_nn.pdf')
        plt.savefig(self.args.directory+name_posterior+'_'+str(testindex)+'_'+''.join(map(str, spirange))+'_paper_im_plot_nn.pdf',bbox_inches='tight')
        plt.close(fig3)

        plt.clf()
        plt.hist(maxloss,bins=1000)        
        plt.xscale('log')
        plt.savefig(self.args.directory+name_posterior+'_'+str(testindex)+'_'+''.join(map(str, spirange))+'_hist_im_plot_nn.pdf',bbox_inches='tight')


if __name__ == "__main__":

    myflow = bayes_inversion()
    # myflow.create_database(spectral_range=5, tauvalues = 9, noise=0.0, batch_size=100)
    # myflow.train_network(num_epochs=4000,continueTraining=False,learning_rate = 1e-4,name_posterior= 'forwardnet_v6C',mhidden_features=370,num_blocks=10,mse=True,typemodel=2)
    
    myflow.test_plots_paper(name_posterior = 'forwardnet_v6C',tauvalues = 9,spirange=[5])

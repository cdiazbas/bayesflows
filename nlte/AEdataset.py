# -*- coding: utf-8 -*-
__author__ = 'carlos.diaz'
# Autoencoder for the output data based on residual networks

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import time
import os
import nde_utils
import nde_ae
from tqdm import tqdm
import sys
# from ipdb import set_trace as stop
# stop()



class bayes_inversion(object):

    # =========================================================================
    def __init__(self, directory = 'dataset_ae_final/', device = 'cpu'):

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
        lines = np.load(mdir+'trainfixe_lines.npy')#[:100,:]
        values = np.load(mdir+'trainfixe_values.npy')#[:100,:]


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
        self.train_loader = nde_utils.basicLoader(lines[wholedataset[:train_split],:], values[wholedataset[:train_split],:], noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)

        self.vali_loader = nde_utils.basicLoader(lines[wholedataset[train_split:],:], values[wholedataset[train_split:],:], noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)

        print("[INFO] len(ltau):", self.lentau)
        print("[INFO] len(waves):", self.lenwave)
        print('[INFO] Datasize obsdata: ',lines.shape)
        print('[INFO] Datasize params:  ',values.shape)
        print('[INFO] Train/valid split: ',train_split,int(lines.shape[0]*(1.0-split)))

        #vali cube:
        print('[INFO] Reading test database')

        mdir = '../gaussian_model/'
        lines = np.load(mdir+'test_lines_exp.npy')
        values = np.load(mdir+'test_values_exp.npy')
        self.test_loader = nde_utils.basicLoader(lines, values, noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)


    # =========================================================================
    def train_network(self, num_epochs = 2000, learning_rate = 1e-6, log_interval = 1, continueTraining=True, name_posterior= 'posterior',num_flows=5,num_blocks=1,mhidden_features=32,modeltype=None, l_size=15):
        

        name_posterior = name_posterior+'_sp'+str(self.spectral_range)
        self.args.y_size = self.lentau*3
        self.args.x_size = self.lenwave
        self.args.l_size = l_size
        if modeltype is None:
            self.model = nde_ae.AE(self.args.y_size, self.args.l_size, train_loader=self.train_loader, hidden_size = [128, 128, 128, 128, 128])
        elif modeltype == 'RAE':
            self.model = nde_ae.RAE(self.args.y_size, self.args.l_size, train_loader=self.train_loader,hidden_size = 64,num_blocks=5)
        else:
            print('no type')
            
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
        
        # Force better fit in chromosphere
        ww = torch.ones(self.args.y_size)
        # 0-8; 9-17; 18-26
        ww[1] = 2.0
        ww[2:6] = 5.0
        ww[3] = 10.0
        ww[4] = 10.0
        ww[5] = 10.0
        ww[6] = 10.0
        ww[7] = 10.0
        ww[10] = 2.0
        ww[11:15] = 5.0
        ww[15] = 10.0
        ww[16] = 10.0
        ww[19] = 2.0
        ww[20:24] = 5.0
        ww[24] = 10.0
        ww[25] = 10.0
        # ww = torch.from_numpy(1/np.load(self.args.directory+'meandiff_inv.npy'))        
        ww[0:9] /= 40.0
        ww[9:18] /= 1.0
        ww[18:] /= 20.0
        # print(ww[0:9])
        # print(ww[9:18])
        # print(ww[18:])

        from tqdm import trange
        t = trange(num_epochs, desc='', leave=True)
        self.valimin = 1e3
        self.count = 0
        self.maxiloop = 100
        for epoch in t:
            self.model.train()
            avgloss = 0
            for batch_idx, (params, data) in enumerate(tqdm(self.train_loader, desc='', leave=False)):
                data = data.to(self.device)
                params = params.to(self.device)
                optimizer.zero_grad()
                loss  = self.model.forward(params,ww)
                loss.backward()
                optimizer.step()
                avgloss += loss.item()
                
            avgloss /= (batch_idx +1)
            train_loss_avg.append(avgloss)


            self.model.eval()
            avgloss2 = 0
            for batch_idx, (params, data) in enumerate(self.vali_loader):
                data = data.to(self.device)
                params = params.to(self.device)
                loss  = self.model.forward(params,ww)
                # loss  = self.model.forward(params)
                avgloss2 += loss.item()
                
            avgloss2 /= (batch_idx +1)
            vali_loss_avg.append(avgloss2)

            argminiv = np.argmin(vali_loss_avg)
            miniv = np.mean(vali_loss_avg[argminiv-1:argminiv+1+1])
            if argminiv == 0:
                miniv = vali_loss_avg[argminiv]

            fig = plt.figure(); plt.plot(train_loss_avg); plt.plot(vali_loss_avg)
            plt.axhline(np.mean(train_loss_avg[-10:]),color='C0',ls='--')
            plt.axhline(np.mean(train_loss_avg[-10:]),color='k',ls='--',alpha=0.5)
            plt.axvline(argminiv,color='k',ls='--',alpha=0.5)
            plt.axhline(miniv,color='C1',ls='--')
            plt.axhline(miniv,color='k',ls='--',alpha=0.5)
            plt.title('loss_final: {0:.3e} / {1:.3e}'.format( np.mean(train_loss_avg[-10:]), miniv ))
            plt.xlabel('Epochs'); plt.ylabel('Loss')
            plt.yscale('log')
            plt.savefig(self.args.directory+self.args.name_posterior+'_train_loss_avg.pdf'); plt.close(fig)

            self.test_plots(8160)

            t.set_postfix({'loss': '{:.3e}'.format(avgloss)})
            t.refresh()

            if avgloss2 < self.valimin:
                self.valimin = np.copy(avgloss2)
                self.count = 0
                torch.save(self.model, self.args.directory+self.args.name_posterior+'_best.pth')
            else:
                self.count += 1
            
            if self.count > self.maxiloop:
                print('[INFO] Done')
                print('[INFO] name_posterior: ',name_posterior)
                sys.exit()



    # =========================================================================
    def test_plots(self, testindex=0,nsamples = 1000):
        import mathtools as mt

        mltau = self.mltau
        waves = self.waves
        testvalue = self.test_loader.dataset.modelparameters[testindex,:]
        testobs = self.test_loader.dataset.observations[testindex,:]

        samples_histo = self.model.sample(testvalue).data.cpu().numpy()
        samples_temp = samples_histo[0,self.lentau*0:self.lentau*1]
        samples_vlos = samples_histo[0,self.lentau*1:self.lentau*2]
        samples_vturb = samples_histo[0,self.lentau*2:self.lentau*3]

        fig3 = plt.figure(figsize=(8,16))
        plt.subplot(411)
        plt.plot(mltau,samples_temp,'.--',color='C1',label='fit (sigma 1&2)')
        plt.plot(mltau,testvalue[self.lentau*0:self.lentau*1], "k", marker='s', markersize=2, label="truth", ls='none')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel("T [kK]");
        plt.ylim(3.0,15.0)
        plt.legend(fontsize=14)
        
        plt.subplot(412)
        plt.plot(mltau,samples_vlos,'.--',color='C1',label='fit (sigma 1&2)')
        plt.plot(mltau,testvalue[self.lentau*1:self.lentau*2], "k", marker='s', markersize=2, label="truth", ls='none')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel(r"v$_{LOS}$ [km/s]");
        plt.ylim(-12.0,+12.0)
        plt.legend(fontsize=14)

        plt.subplot(413)
        plt.plot(mltau,samples_vturb,'.--',color='C1',label='fit (sigma 1&2)')
        plt.plot(mltau,testvalue[self.lentau*2:self.lentau*3], "k", marker='s', markersize=2, label="truth", ls='none')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel(r"v$_{TURB}$ [km/s]");
        plt.ylim(0.0,+10.0)
        plt.legend(fontsize=14)        
        
        plt.legend(fontsize=14)
        plt.savefig(self.args.directory+self.args.name_posterior+'_'+str(testindex)+'_im_plot_nn.pdf')
        plt.close(fig3)




    # =========================================================================
    def test_error(self, nsamples = 10000, name_posterior = 'posterior',tauvalues = 9,spirange=[0,1],testindex=11387,gotostic = False):
        import matplotlib
        matplotlib.rcParams['axes.formatter.useoffset'] = False
        
        inc = 0.8
        fig3 = plt.figure(figsize=(8*inc,14*inc))
        
        name_posterior = name_posterior+'_sp'+str(self.spectral_range)
        self.model = torch.load(self.args.directory+name_posterior+'_best.pth')

        mltau = self.mltau
        waves = self.waves
        ntestindex = 10000
        listdiff = []
        for testindex in tqdm(range(ntestindex)): 
            testvalue = self.train_loader.dataset.modelparameters[testindex,:]
            samples_histo = self.model.sample(testvalue).data.cpu().numpy()
            samples_temp = samples_histo[0,self.lentau*0:self.lentau*1]
            samples_vlos = samples_histo[0,self.lentau*1:self.lentau*2]
            samples_vturb = samples_histo[0,self.lentau*2:self.lentau*3]
            absdiff = np.abs(samples_histo[0,:] - testvalue)
            listdiff.append(absdiff)

        meandiff = np.mean(np.array(listdiff),axis=0)
        nmeandiff = np.copy(meandiff)
        # nmeandiff[0:9] *= 50.0
        # nmeandiff[9:18] *= 1.0
        # nmeandiff[18:] *= 10.0

        fig3 = plt.figure(figsize=(8,16))
        plt.subplot(411)
        plt.plot(mltau,meandiff[self.lentau*0:self.lentau*1], '.--',color='C1')
        plt.plot(mltau,nmeandiff[self.lentau*0:self.lentau*1], '.--',color='C2')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel("T [kK]");
        plt.ylim(0.0,0.6)
        
        plt.subplot(412)
        plt.plot(mltau,meandiff[self.lentau*1:self.lentau*2], '.--',color='C1')
        plt.plot(mltau,nmeandiff[self.lentau*1:self.lentau*2], '.--',color='C2')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel(r"v$_{LOS}$ [km/s]");
        plt.ylim(0.0,0.6)

        plt.subplot(413)
        plt.plot(mltau,meandiff[self.lentau*2:self.lentau*3], '.--',color='C1')
        plt.plot(mltau,nmeandiff[self.lentau*2:self.lentau*3], '.--',color='C2')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel(r"v$_{TURB}$ [km/s]");
        plt.ylim(0.0,+0.6)

        plt.savefig(self.args.directory+name_posterior+'_im_plot_error.pdf')
        plt.close(fig3)




if __name__ == "__main__":

    myflow = bayes_inversion()
    myflow.create_database(spectral_range=5, tauvalues = 9, noise=1e-2)
    # myflow.train_network(num_epochs=3000,continueTraining=False,learning_rate = 1e-5,name_posterior= 'encoder_5_5_64e_20',modeltype='RAE',l_size=20)

    myflow.test_error(name_posterior= 'encoder_5_5_64e_20')


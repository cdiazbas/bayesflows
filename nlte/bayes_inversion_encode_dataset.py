# -*- coding: utf-8 -*-
__author__ = 'carlos.diaz'
# Density estimation in spectropolarimetric inversions (using AE in dataset) using GPU if available

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import time
import os
import nde_utils
import nde_nflow
from tqdm import tqdm
import sys
# from ipdb import set_trace as stop



class bayes_inversion(object):

    # =========================================================================
    def __init__(self, directory = 'bayes_inversion_output_final/', device = None):

        # Configuration
        self.args = nde_utils.dotdict()
        self.args.directory = directory
        
        self.cudaOption = torch.cuda.is_available()
        print('[INFO] Cuda:', self.cudaOption)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is not None: #forcing the device
            self.device = device

        print('[INFO] Device:', self.device)

        self.args.kwargs = {'num_workers': 1, 'pin_memory': True} if self.device=="cuda" else {}

        if not os.path.exists(self.args.directory): os.makedirs(self.args.directory)

        self.output_decoder ='dataset_ae_final/encoder_5_5_64e_20_sp5_best_.pth'
        self.decoder = torch.load(self.output_decoder)

    # =========================================================================
    def create_database(self, batch_size = 100, tauvalues = 15, spectral_range=0, noise=5e-4,size=1e4):

        import sparsetools as sp
        print('[INFO] Using spectral range '+str(spectral_range))
        print('[INFO] Reading database')

        mdir = '../gaussian_model/'
        lines = np.load(mdir+'trainfixe_lines.npy')[:int(size),:]
        values = np.load(mdir+'trainfixe_values.npy')[:int(size),:]


        self.waves_info = np.load(mdir+'train_waves_info.npy')
        self.waves = np.load(mdir+'train_waves.npy')
        self.lenwave = len(self.waves)
        self.spectral_range = spectral_range
        if  self.spectral_range == 5:
            spc_idx = range(self.lenwave)
        elif self.spectral_range == 0:
            spc_idx = range(21,self.lenwave )

        self.ltau = np.load(mdir+'train_ltau.npy')
        self.mltau = np.load(mdir+'train_mltau.npy')
        self.lentau = len(self.mltau)
        self.spectral_idx = np.load(mdir+'train_spectral_idx.npy')

        self.waves_info = self.waves_info[spc_idx]
        self.waves = self.waves[spc_idx]
        self.lenwave = len(self.waves)
        self.spectral_idx = self.spectral_idx[spc_idx]
        lines = lines[:,spc_idx]

        time0 = time.time()
        values = self.decoder.sample_encoder(values)
        self.latent_size = values.shape[1]
        print('Latent [s]: {:.2f}'.format(time.time()-time0))

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
        lines = np.load(mdir+'test_lines_exp.npy')[:,spc_idx]
        values = np.load(mdir+'test_values_exp.npy')
        self.test_loader = nde_utils.basicLoader(lines, values, noise=noise, batch_size=self.args.batch_size, shuffle=True, **self.args.kwargs)


    # =========================================================================
    def train_network(self, num_epochs = 2000, learning_rate = 1e-6, log_interval = 1, continueTraining=True, name_posterior= 'posterior',num_flows=5,num_blocks=1,mhidden_features=32,transformtype="rq-coupling"):
        

        name_posterior = name_posterior+'_sp'+str(self.spectral_range)
        self.args.y_size = self.lentau*3

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        self.args.y_size = self.latent_size # Latent dimension
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        self.args.x_size = self.lenwave
        self.model = nde_nflow.NFLOW(self.args.y_size, self.args.x_size,num_flows=num_flows, mhidden_features=mhidden_features, num_blocks=num_blocks, 
                    train_loader=self.train_loader, embedding_net=None, transformtype=transformtype, output_decoder=self.output_decoder )
        
        nde_utils.get_params(self.model)

        self.args.learning_rate = learning_rate
        self.args.num_epochs = num_epochs
        self.args.log_interval = log_interval
        self.args.name_posterior = name_posterior
        print('[INFO] name_posterior: ',name_posterior)

        if continueTraining: self.model = torch.load(self.args.directory+name_posterior+'_best.pth'); print('Loading previous weigths...')

        #@@@@@@@@@@@ CUDA @@@@@@@@@@@@@@@@
        if self.cudaOption:
            print('[INFO] Cuda model:',torch.cuda.get_device_name(0))
            self.model.cuda()
            print('[INFO] Parameters in GPU: ',next(self.model.parameters()).is_cuda)            
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        train_loss_avg = []
        vali_loss_avg = []
        time0 = time.time()
        

        from tqdm import trange
        t = trange(num_epochs, desc='', leave=True)
        self.valimin = 1e3
        self.count = 0
        self.maxiloop = 100
        self.lastsaved = 0
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
                
            avgloss /= (batch_idx +1)
            train_loss_avg.append(avgloss)

            self.model.eval()
            avgloss2 = 0
            for batch_idx, (params, data) in enumerate(self.vali_loader):
                data = data.to(self.device)
                params = params.to(self.device)
                loss  = self.model.forward(params, data)
                avgloss2 += loss.item()
                
            avgloss2 /= (batch_idx +1)
            vali_loss_avg.append(avgloss2)

            argminiv = np.argmin(vali_loss_avg)
            miniv = np.mean(vali_loss_avg[argminiv-1:argminiv+1+1])
            ministd = 2*np.std(vali_loss_avg[argminiv-1:argminiv+1+1])

            fig = plt.figure(); plt.plot(train_loss_avg); plt.plot(vali_loss_avg)
            plt.axhline(np.mean(train_loss_avg[-10:]),color='C0',ls='--')
            plt.axhline(np.mean(train_loss_avg[-10:]),color='k',ls='--',alpha=0.5)
            # plt.axhline(np.mean(vali_loss_avg[-10:]),color='C1',ls='--')
            # plt.axhline(np.mean(vali_loss_avg[-10:]),color='k',ls='--',alpha=0.5)
            # plt.axhline(np.min(vali_loss_avg[:]),color='C1',ls='--')
            # plt.axhline(np.min(vali_loss_avg[:]),color='k',ls='--',alpha=0.5)
            plt.axvline(argminiv,color='k',ls='--',alpha=0.5)
            plt.axhline(miniv,color='C1',ls='--')
            plt.axhline(miniv,color='k',ls='--',alpha=0.5)
            if avgloss2 < self.valimin+ ministd:
                plt.axvline(epoch,color='k',ls='--',alpha=0.2)
            else:
                plt.axvline(self.lastsaved,color='k',ls='--',alpha=0.2)
            plt.title('loss_final: {0:2.2f} / {1:2.2f}'.format( np.mean(train_loss_avg[-10:]), miniv ))
            plt.xlabel('Epochs'); plt.ylabel('Loss')
            plt.savefig(self.args.directory+self.args.name_posterior+'_train_loss_avg.pdf'); plt.close(fig)

            self.test_plots(8160)

            t.set_postfix({'loss': '{:.2f}'.format(avgloss)})
            t.refresh()

            if avgloss2 < self.valimin+ ministd:
                self.lastsaved = epoch
                if avgloss2 < self.valimin: self.valimin = np.copy(avgloss2)
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

        samples_histo = self.model.obtain_samples(testobs,nsamples).data.cpu().numpy()
        samples_temp = samples_histo[:,self.lentau*0:self.lentau*1]
        samples_vlos = samples_histo[:,self.lentau*1:self.lentau*2]
        samples_vturb = samples_histo[:,self.lentau*2:self.lentau*3]


        fig3 = plt.figure(figsize=(8,16))
        plt.subplot(411)
        plt.fill_between(mltau,np.percentile(samples_temp, 2.5, axis=0),np.percentile(samples_temp, 97.5, axis=0),alpha=0.2,color='C1')
        plt.fill_between(mltau,np.percentile(samples_temp, 16, axis=0),np.percentile(samples_temp, 84, axis=0),alpha=0.4,color='C1')
        plt.plot(mltau,np.percentile(samples_temp, 50, axis=0),'.--',color='C1',label='fit (sigma 1&2)')
        plt.plot(mltau,testvalue[self.lentau*0:self.lentau*1], "k", marker='s', markersize=2, label="truth", ls='none')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel("T [kK]");
        plt.ylim(3.0,15.0)
        plt.legend(fontsize=14)
        
        plt.subplot(412)
        plt.fill_between(mltau,np.percentile(samples_vlos, 2.5, axis=0),np.percentile(samples_vlos, 97.5, axis=0),alpha=0.2,color='C1')
        plt.fill_between(mltau,np.percentile(samples_vlos, 16, axis=0),np.percentile(samples_vlos, 84, axis=0),alpha=0.4,color='C1')
        plt.plot(mltau,np.percentile(samples_vlos, 50, axis=0),'.--',color='C1',label='fit (sigma 1&2)')
        plt.plot(mltau,testvalue[self.lentau*1:self.lentau*2], "k", marker='s', markersize=2, label="truth", ls='none')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel(r"v$_{LOS}$ [km/s]");
        plt.ylim(-12.0,+12.0)
        plt.legend(fontsize=14)

        plt.subplot(413)
        plt.fill_between(mltau,np.percentile(samples_vturb, 2.5, axis=0),np.percentile(samples_vturb, 97.5, axis=0),alpha=0.2,color='C1')
        plt.fill_between(mltau,np.percentile(samples_vturb, 16, axis=0),np.percentile(samples_vturb, 84, axis=0),alpha=0.4,color='C1')
        plt.plot(mltau,np.percentile(samples_vturb, 50, axis=0),'.--',color='C1',label='fit (sigma 1&2)')
        plt.plot(mltau,testvalue[self.lentau*2:self.lentau*3], "k", marker='s', markersize=2, label="truth", ls='none')
        plt.xlabel(r"log($\tau$)")
        plt.ylabel(r"v$_{TURB}$ [km/s]");
        plt.ylim(0.0,+10.0)
        plt.legend(fontsize=14)        
        
        plt.subplot(414)
        plt.plot(waves, testobs,'.--',color='C1',label='Full line')
        plt.plot(waves, testobs, "k", marker='s', markersize=5, label="Used points", ls='none')
        plt.xlabel(r"$\lambda - \lambda_0 [\AA]$")
        plt.ylabel(r"I/I$_{C(QS)}$");
        plt.legend(fontsize=14)
        plt.savefig(self.args.directory+self.args.name_posterior+'_'+str(testindex)+'_im_plot_nn.pdf')
        plt.close(fig3)



if __name__ == "__main__":

    myflow = bayes_inversion()
    myflow.create_database(spectral_range=5, tauvalues = 9, noise=1e-2, size=1e4)
    myflow.train_network(num_epochs=3000,continueTraining=False,learning_rate = 1e-4,name_posterior= 'qposterior_encoding_15_10_64e_t9_1e-2g_1e6',num_flows=15,num_blocks=10,mhidden_features=64)


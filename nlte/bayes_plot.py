# -*- coding: utf-8 -*-
__author__ = 'carlos.diaz'
# Routine with plot utils

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
from ipdb import set_trace as stop

# =========================================================================
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot
    """
    from mpl_toolkits import axes_grid1
    from matplotlib.pyplot import gca, sca
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
    
# ====================================================================
import matplotlib.colors as mcolors
def make_colormap(seq):
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

c = mcolors.ColorConverter().to_rgb
# vcolormap = make_colormap([c('darkred'),c('#ff7f0e'), 0.5, c('#1f77b4'),c('black')])

# =========================================================================
def fix_leakage(prior,samples, logprob=None):
    # Remove samples outside the prior
    index_param = []
    for i_param in range(samples.shape[1]):
        index_param.append( np.where(samples[:,i_param]<prior[0][i_param])[0] )
        index_param.append( np.where(samples[:,i_param]>prior[1][i_param])[0] )

    import itertools
    final_index = np.array(list(itertools.chain.from_iterable(index_param)))
    if len(final_index) > 0:
        # print(samples[final_index,:])
        samples = np.delete(samples,final_index,axis=0)
        if logprob is not None:
            logprob = np.delete(logprob,final_index,axis=0)
            return samples, logprob
    
    if len(final_index) < 1 and  logprob is not None:
        return samples, logprob
    
    if logprob is None:
        return samples


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
    def create_database(self, batch_size = 100, tauvalues = 15, spectral_range=0, noise=5e-4,size=1e6):

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


        split = 0.9
        train_split = int(lines.shape[0]*split)
        wholedataset = np.arange(lines.shape[0])
        np.random.shuffle(wholedataset)

        mdd = np.ones(27,dtype=np.float32)*1e-2

        self.args.batch_size = batch_size
        self.train_loader = nde_utils.basicLoader(lines[wholedataset[:train_split],:], values[wholedataset[:train_split],:], noise=noise, batch_size=self.args.batch_size, shuffle=True, xnoise=1.0,amplitude=mdd, **self.args.kwargs)

        self.vali_loader = nde_utils.basicLoader(lines[wholedataset[train_split:],:], values[wholedataset[train_split:],:], noise=noise, batch_size=self.args.batch_size, shuffle=True, xnoise=1.0,amplitude=mdd, **self.args.kwargs)

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
    def test_plots_paper(self, nsamples = 10000, name_posterior = 'posterior',tauvalues = 9,spirange=[0,5],testindex=11387,gotostic = False):
        import matplotlib
        import mathtools as mt

        import warnings
        warnings.filterwarnings("ignore")
        matplotlib.rcParams['axes.formatter.useoffset'] = False
        
        inc = 0.8
        fig3 = plt.figure(figsize=(8*inc,14*inc))

        colorsp = ['C0','C4','k']
        colorsp = ['C1','C5','k']
        lsi = ['-','--','.-']
        cc = 0
        mlabel = ['[Fe I]','[Ca II + Fe I]']


        for spi in spirange:
            self.create_database(tauvalues = tauvalues, spectral_range=spi)
            self.args.y_size = self.lentau*3
            self.args.x_size = self.lenwave
            self.model = torch.load(self.args.directory+name_posterior+'_sp'+str(self.spectral_range)+'_best.pth')


            mltau = self.mltau
            mltau[0] -= 0.05
            mltau[-1] += 0.05
            self.ltau[0] -= 0.05
            self.ltau[-1] += 0.05

            waves = self.waves
            testvalue = self.test_loader.dataset.modelparameters[testindex,:]
            testobs = self.test_loader.dataset.observations[testindex,:]
            time0 = time.time()
            samples_histo, logprob = self.model.sample_and_log_prob(testobs,nsamples,extranoise=1e-2,batch_size=nsamples)
            print(f'Samples per second in {name_posterior}:',int(nsamples/(time.time()-time0)))

            true_temp = testvalue[self.lentau*0:self.lentau*1]
            true_vlos = testvalue[self.lentau*1:self.lentau*2]
            true_vturb = testvalue[self.lentau*2:self.lentau*3]

            samples_temp = samples_histo[:,self.lentau*0:self.lentau*1]
            samples_vlos = samples_histo[:,self.lentau*1:self.lentau*2]
            samples_vturb = samples_histo[:,self.lentau*2:self.lentau*3]

            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print(samples_histo.shape,logprob.shape) # 68,95,99.7
            prior = [np.percentile(samples_histo, 50-95/2., axis=0),np.percentile(samples_histo, 50+95/2., axis=0)]
            samples_histo, logprob = fix_leakage(prior,samples_histo, logprob)
            map_ = np.argmax(+logprob)
            map_args = samples_histo[map_,:]
            print(samples_histo.shape,logprob.shape)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


            resampling = False
            if resampling is True:
                try:
                    print('+++++++++++++++++++++++++++++++++++')
                    print('+++++++++++++++++++++++++++++++++++')
                    # Synthesis
                    name_forward='forwardnet_v4'
                    self.forward = torch.load(self.args.directory+name_forward+'_sp'+str(self.spectral_range)+'_best.pth')
                    stokes_flow  = self.forward.evaluate(torch.tensor(samples_histo))

                    # Importance reweighting
                    sigma2 = (1e-2)**2.
                    ntestobs = testobs*1.0 + np.random.normal(0.,1e-2,size=len(testobs))
                    new_logprob = -0.5 * np.sum((ntestobs - stokes_flow) ** 2 / sigma2 + np.log(sigma2) + np.log(2*np.pi),axis=1)
                    # samples_histo, logprob = nde_utils.importance_reweighting(samples_histo, logprob, new_logprob)
                    # samples_histo, logprob = nde_utils.importance_reweighting(samples_histo, logprob, new_logprob, linear=True)

                    samples_temp = samples_histo[:,self.lentau*0:self.lentau*1]
                    samples_vlos = samples_histo[:,self.lentau*1:self.lentau*2]
                    samples_vturb = samples_histo[:,self.lentau*2:self.lentau*3]
                    map_ = np.argmax(+logprob)
                    map_args = samples_histo[map_,:]
                    print('+++++++++++++++++++++++++++++++++++')
                    print('+++++++++++++++++++++++++++++++++++')
                except:
                    pass


            plt.subplot(411)
            if cc == 0 : plt.plot(mltau,true_temp, "k", marker='s', markersize=2, label="Original values", ls='none')



            plt.fill_between(self.ltau,mt.bezier3(self.mltau, np.percentile(samples_temp, 16, axis=0), self.ltau),mt.bezier3(self.mltau, np.percentile(samples_temp, 84, axis=0), self.ltau),alpha=0.4,color=colorsp[cc])
            plt.plot(self.ltau,mt.bezier3(self.mltau, np.percentile(samples_temp, 50, axis=0), self.ltau),lsi[cc],color=colorsp[cc],label=r'Median $\pm\sigma$ '+mlabel[cc])
            plt.tick_params(axis='x',labelbottom=False)
            plt.xlim(-7.05,+1.0)
            plt.ylabel("T [kK]");
            plt.ylim(3.0,14.0)
            plt.legend(fontsize=10,loc="upper right")
            
            plt.subplot(412)
            if cc == 0 :plt.plot(mltau,true_vlos, "k", marker='s', markersize=2, label="truth", ls='none')
            plt.fill_between(self.ltau,mt.bezier3(self.mltau, np.percentile(samples_vlos, 16, axis=0), self.ltau),mt.bezier3(self.mltau, np.percentile(samples_vlos, 84, axis=0), self.ltau),alpha=0.4,color=colorsp[cc])
            plt.plot(self.ltau,mt.bezier3(self.mltau, np.percentile(samples_vlos, 50, axis=0), self.ltau),lsi[cc],color=colorsp[cc],label=r'Median $\pm\sigma$ '+mlabel[cc])
            plt.ylabel(r"v$_{LOS}$ [km/s]");
            plt.ylim(-12.0,+12.0)
            plt.xlim(-7.05,+1.0)
            plt.tick_params(axis='x',labelbottom=False)

            plt.subplot(413)
            if cc == 0 :plt.plot(mltau,true_vturb, "k", marker='s', markersize=2, label="truth", ls='none')
            plt.fill_between(self.ltau,mt.bezier3(self.mltau, np.percentile(samples_vturb, 16, axis=0), self.ltau),mt.bezier3(self.mltau, np.percentile(samples_vturb, 84, axis=0), self.ltau),alpha=0.4,color=colorsp[cc])
            plt.plot(self.ltau,mt.bezier3(self.mltau, np.percentile(samples_vturb, 50, axis=0), self.ltau),lsi[cc],color=colorsp[cc],label=r'Median $\pm\sigma$ '+mlabel[cc])
            plt.xlabel(r"log($\tau_{500}$)")
            plt.ylabel(r"v$_{TURB}$ [km/s]");
            plt.ylim(0.0,+12.0)
            plt.xlim(-7.05,+1.0)


            if gotostic is True:        
                # To STiC model:
                synthep =  self.nn2stic(samples_histo,map_, testvalue,self.args.directory+name_posterior+'_'+str(testindex)+'_'+''.join(map(str, spirange)) )
                mwav = synthep.wav

            ax7 = plt.subplot(427)
            splitdx = 90
            plt.errorbar(self.waves_info[waves<splitdx]-8542.1, testobs[waves<splitdx], ls='none', yerr=1e-2,color='k',zorder=-3, marker='s', markersize=2)
            plt.xlabel(r"$\lambda - 8542.1$ $[\rm \AA]$")
            plt.ylabel(r"I/I$_{C}$");
            if gotostic is True and self.spectral_range == 5:#len(testobs[waves<130]) > 2: 
                plt.plot(mwav[(mwav>8000) & (mwav<8550)]-8542.1,synthep.dat[0,0,0,:,0][(mwav>8000) & (mwav<8550)],lsi[cc],color=colorsp[cc],label=mlabel[cc])
            else:
                plt.plot(self.waves_info[waves<splitdx]-8542.1, testobs[waves<splitdx],lsi[cc],color=colorsp[cc],label=mlabel[cc])


            ax8 = plt.subplot(428)
            if cc ==0: plt.errorbar(self.waves_info[waves>splitdx]- 6301.5, testobs[waves>splitdx], ls='none', yerr=1e-2,color='k',zorder=-3, marker='s', markersize=2,label='Original values')
            if gotostic is True: 
                plt.plot(mwav[(mwav>6300) & (mwav<6302)]- 6301.5,synthep.dat[0,0,0,:,0][(mwav>6300) & (mwav<6302)],lsi[cc],color=colorsp[cc],label=mlabel[cc])
            else:
                plt.plot(self.waves_info[waves>splitdx]- 6301.5, testobs[waves>splitdx],lsi[cc],color=colorsp[cc],label=mlabel[cc])

            plt.xlabel(r"$\lambda - 6301.5$ $[\rm \AA]$")
            plt.legend(fontsize=10,loc="upper right")
            cc += 1


        if gotostic is True: 
            print('==============')
            print('std Ca',np.std(testobs[waves<splitdx]-synthep.dat[0,0,0,self.spectral_idx,0][waves<splitdx]))
            print('std Fe',np.std(testobs[waves>splitdx]-synthep.dat[0,0,0,self.spectral_idx,0][waves>splitdx]))
            print('max Ca',np.max(np.abs(testobs[waves<splitdx]-synthep.dat[0,0,0,self.spectral_idx,0][waves<splitdx])))
            print('max Fe',np.max(np.abs(testobs[waves>splitdx]-synthep.dat[0,0,0,self.spectral_idx,0][waves>splitdx])))


        pos7 = ax7.get_position()
        pos8 = ax8.get_position()
        ax7.set_position([0.125,0.08,0.34,0.18])
        ax8.set_position([0.5477272727272726+0.01, 0.08, 0.34, 0.18])

        print(self.args.directory+name_posterior+'_'+str(testindex)+'_'+''.join(map(str, spirange))+'_paper_im_plot_nn.pdf')
        plt.savefig(self.args.directory+name_posterior+'_'+str(testindex)+'_'+''.join(map(str, spirange))+'_paper_im_plot_nn.pdf',bbox_inches='tight')
        
        plt.clf()
        plt.figure()
        mn = np.median(samples_histo.T, axis=1)
        std = np.std(samples_histo.T, axis=1)
        corr_matrix = np.cov(samples_histo.T / std[:, None])
        ax = plt.imshow(corr_matrix,cmap='RdBu',vmin=-1,vmax=+1)
        cbar = add_colorbar(ax,30)
        cbar.set_label("Covariance matrix")
        plt.axvline(9-0.5,color='k',ls='--')
        plt.axhline(9-0.5,color='k',ls='--')
        plt.axvline(18-0.5,color='k',ls='--')
        plt.axhline(18-0.5,color='k',ls='--')
        plt.xlabel("          T [kK]                 "+r"v$_{LOS}$ [km/s]          "+r"v$_{TURB}$ [km/s]   ");
        plt.ylabel(r"v$_{TURB}$ [km/s]          "+r"v$_{LOS}$ [km/s]          "+"   T [kK]");
        plt.xticks(np.arange(0, 27, step=3))
        plt.yticks(np.arange(0, 27, step=3))
        plt.savefig(self.args.directory+name_posterior+'_'+str(testindex)+'_'+''.join(map(str, spirange))+'_paper_cov.pdf',bbox_inches='tight')
        print(self.args.directory+name_posterior+'_'+str(testindex)+'_'+''.join(map(str, spirange))+'_paper_cov.pdf')



    # =========================================================================
    def nn2stic(self, samples_histo, map_, testvalue,name):
        import mathtools as mt
        import sparsetools as sp

        map_args = samples_histo[map_,:]
        samples_temp = samples_histo[:,self.lentau*0:self.lentau*1]
        samples_vlos = samples_histo[:,self.lentau*1:self.lentau*2]
        samples_vturb = samples_histo[:,self.lentau*2:self.lentau*3]
        true_temp = testvalue[self.lentau*0:self.lentau*1]
        true_vlos = testvalue[self.lentau*1:self.lentau*2]
        true_vturb = testvalue[self.lentau*2:self.lentau*3]
        mltau = self.mltau
        waves = self.waves

        nx = 1; ny = 1; nt = 1
        ndep = len(self.ltau)
        model_stic = sp.model(nx=nx, ny=ny, ndep=ndep, nt=nt)

        for jj in tqdm(range(nx)):
            for kk in range(ny):
                # Median value
                # model_stic.temp[0,jj,kk,:] = mt.bezier3(self.mltau, np.percentile(samples_temp, 50, axis=0), self.ltau)*1000.0
                # model_stic.vlos[0,jj,kk,:] = mt.bezier3(self.mltau, np.percentile(samples_vlos, 50, axis=0), self.ltau)*1e5
                # model_stic.vturb[0,jj,kk,:] = mt.bezier3(self.mltau, np.percentile(samples_vturb, 50, axis=0), self.ltau)*1e5
                
                # MAP value
                samples_temp = map_args[self.lentau*0:self.lentau*1]
                samples_vlos = map_args[self.lentau*1:self.lentau*2]
                samples_vturb = map_args[self.lentau*2:self.lentau*3]
                model_stic.temp[0,jj,kk,:] = mt.bezier3(self.mltau, samples_temp, self.ltau)*1000.0
                model_stic.vlos[0,jj,kk,:] = mt.bezier3(self.mltau, samples_vlos, self.ltau)*1e5
                model_stic.vturb[0,jj,kk,:] = mt.bezier3(self.mltau, samples_vturb, self.ltau)*1e5

                # Originals to STiC for comparison
                # model_stic.temp[0,jj,kk,:] = mt.bezier3(self.mltau, true_temp, self.ltau)*1000.0
                # model_stic.vlos[0,jj,kk,:] = mt.bezier3(self.mltau, true_vlos, self.ltau)*1e5
                # model_stic.vturb[0,jj,kk,:] = mt.bezier3(self.mltau, true_vturb, self.ltau)*1e5

        model_stic.ltau[0,:,:,:] = self.ltau 
        model_stic.pgas[0,:,:,:] = 1.0 
        model_stic.Bln[0,:,:,:] = 0.0
        model_stic.Bho[0,:,:,:] = 0.0
        model_stic.azi[0,:,:,:] = 0.0
        model_stic.write('_atmosmodel.nc') #name
        
        # Run Stic:
        retval = os.getcwd()
        os.system('mv _atmosmodel.nc synthe_1profileB/atmos_model.nc')
        os.chdir('synthe_1profileB')
        os.system('mpiexec.openmpi -np 2 ./STiC.x')
        mprofile = sp.profile('synthetic.nc')
        os.chdir(retval) # Previous directory
        return mprofile


    # =========================================================================
    def test_pp_plot(self, nsamples = 1000,name_posterior='posterior',spectral_range=0,directory='.',tauvalues = 15):
        
        self.create_database(tauvalues = tauvalues, spectral_range=spectral_range)
        self.model = torch.load(self.args.directory+name_posterior+'_sp'+str(self.spectral_range)+'_best.pth')

        import mathtools as mt
        from scipy import stats

        mltau = self.mltau
        waves = self.waves
        
        # Perform inference on a large number of injections. This takes a few minutes.
        neval = 1000    # number of injections
        nparams = tauvalues*3

        percentiles = np.empty((neval, nparams))
        for idx in tqdm(range(neval)):
            testindex = np.random.randint(0,36450)
            testvalue = self.train_loader.dataset.modelparameters[testindex,:]
            testobs = self.train_loader.dataset.observations[testindex,:]
            samples_histo = self.model.obtain_samples(testobs,nsamples).data.cpu().numpy()
            for n in range(nparams):
                percentiles[idx, n] = stats.percentileofscore(samples_histo[:,n], testvalue[n])

        parameter_labels = []
        for zz in ['T','vlos','vturb']:
            for jj in range(tauvalues):
                parameter_labels.append(zz+str(jj))

        percentiles = percentiles/100.0
        nparams = percentiles.shape[-1]
        nposteriors = percentiles.shape[0]

        ordered = np.sort(percentiles, axis=0)
        ordered = np.concatenate((np.zeros((1, nparams)), ordered, np.ones((1, nparams))))
        y = np.linspace(0, 1, nposteriors + 2)
            
        fig = plt.figure(figsize=(10,10))
        color = ['C0','C1','C2','C3']
        ltauplot = -3
        mindex = np.argmin(np.abs(self.mltau-ltauplot))

        for n in range(nparams):
            if np.abs(mindex - n%tauvalues) < 1e-1:
                mcolor = n//tauvalues
                plt.step(ordered[:, n], y, where='post', label=parameter_labels[n],color=color[mcolor])
                print(np.max(ordered[:, n]-y))
            
        plt.plot(y, y, 'k--')
        plt.legend()
        plt.ylabel(r'$CDF(p)$')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.xlabel(r'$p$')
        ax = fig.gca()
        ax.set_aspect('equal', anchor='SW')
        plt.savefig(self.args.directory+name_posterior+'_test_pp_plot.pdf',bbox_inches='tight')





    # =========================================================================
    def test_uncertainty2(self, nsamples = 5000, name_posterior = 'posterior',tauvalues = 9,spirange=[0,1],gotostic = False, name_forward='forwardnet_v6C'):
        import matplotlib
        matplotlib.rcParams['axes.formatter.useoffset'] = False

        for spi in spirange:
            self.create_database(tauvalues = tauvalues, spectral_range=spi)
            self.args.y_size = self.lentau*3
            self.args.x_size = self.lenwave
            mltau = self.mltau
            waves = self.waves

            # Forward network
            self.forward = torch.load(self.args.directory+name_forward+'_sp'+str(self.spectral_range)+'_best.pth')

            nprofiles = 100#00#00#2#00
            ncheck = 1000#1000
            nsamples = 2000#10000

            globalmaxprob = []
            globalmaxprob2 = []
            globaldistri = []

            datastr = [
            'posterior_15_10_64e_t9_1e-2g_1e4',
            'posterior_15_10_64e_t9_1e-2g_1e5',
            'posterior_15_10_64e_t9_1e-2g_1e6',
            'qposterior_encoding_15_10_64e_t9_1e-2g_1e4',
            'qposterior_encoding_15_10_64e_t9_1e-2g_1e5',
            'qposterior_encoding_15_10_64e_t9_1e-2g_1e6',            
            'qposterior_encoding_15_10_64e_t9_1e-2g_1e4',
            'qposterior_encoding_15_10_64e_t9_1e-2g_1e5',
            'qposterior_encoding_15_10_64e_t9_1e-2g_1e6', 
            ]


            labelstr = np.arange(len(datastr))
            labelstr = [1e4,1e5,1e6]

            cc = 0
            importance_reweighting_apply = False
            for datastri in tqdm(datastr):
                if cc >5:
                    importance_reweighting_apply = True 
                    pass

                # Inversion
                # self.model = torch.load(self.args.directory+name_posterior+datastri+'_sp'+str(self.spectral_range)+'_best.pth')
                self.model = torch.load(self.args.directory+name_posterior+datastri+'_sp'+str(self.spectral_range)+'_best.pth',map_location=torch.device('cpu'))

                aa = np.array([],dtype=np.float32)
                bb = np.array([],dtype=np.float32)
                for testindex in tqdm(range(nprofiles)): #,leave=False

                    testvalue = self.test_loader.dataset.modelparameters[testindex,:]
                    testobs0 = self.test_loader.dataset.observations[testindex,:]
                    testobs = testobs0*1.0 + np.random.normal(0,1e-2,size=testobs0.shape)
                    time0 = time.time()

                    # samples_histo = self.model.obtain_samples(testobs,nsamples).data.cpu().numpy()
                    samples_histo, logprob = self.model.sample_and_log_prob(testobs.astype('float32'),nsamples,batch_size=nsamples)
                    print(f'Samples per second in {name_posterior+datastri}:',int(nsamples/(time.time()-time0)))

                    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    # print(samples_histo.shape,logprob.shape) # 68,95,99.7
                    prior = [np.percentile(samples_histo, 50-99.7/2., axis=0),np.percentile(samples_histo, 50+99.7/2., axis=0)]
                    samples_histo, logprob = fix_leakage(prior,samples_histo, logprob)
                    # samples_histo, logprob = fix_leakage(prior,samples_histo, logprob)
                    map_ = np.argmax(+logprob)
                    map_args = samples_histo[map_,:]
                    # print(samples_histo.shape,logprob.shape)
                    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')


                    # Synthesis
                    for icheck in range(ncheck):
                        params = samples_histo[icheck,:]
                        syn_flow  = self.forward.evaluate(torch.tensor(params))

                        if (icheck == 0):
                            stokes_flow = np.zeros((ncheck, len(syn_flow)))                
                            samples_flow = np.zeros((ncheck, len(params)))                
                        stokes_flow[icheck, :] = syn_flow
                        samples_flow[icheck, :] = logprob[icheck]
                    # Maxprob
                    syn_flow  = self.forward.evaluate(torch.tensor(map_args))


                    if importance_reweighting_apply is True:
                        print('+++++++++++++++++++++++++++++++++++')
                        print('+++++++++++++++++++++++++++++++++++')

                        # Importance reweighting
                        sigma2 = (1e-2)**2.
                        new_logprob = -0.5 * np.sum((testobs - stokes_flow) ** 2 / sigma2 + np.log(sigma2) + np.log(2*np.pi),axis=1)
                        samples_histo, logprob = nde_utils.importance_reweighting(samples_histo[:ncheck,:], logprob[:ncheck], new_logprob)

                        # Synthesis
                        for icheck in range(ncheck):
                            params = samples_histo[icheck,:]
                            syn_flow  = self.forward.evaluate(torch.tensor(params))

                            if (icheck == 0):
                                stokes_flow = np.zeros((ncheck, len(syn_flow)))                
                            stokes_flow[icheck, :] = syn_flow

                        # Maxprob
                        map_ = np.argmax(+logprob)
                        map_args = samples_histo[map_,:]
                        syn_flow  = self.forward.evaluate(torch.tensor(map_args))
                        print('+++++++++++++++++++++++++++++++++++')
                        print('+++++++++++++++++++++++++++++++++++')

                    aa = np.append(aa,(stokes_flow-testobs).flatten() )
                    bb = np.append(bb,syn_flow-testobs)

                globalmaxprob2.append(np.max(bb))
                globalmaxprob.append(np.std(bb))
                globaldistri.append(np.std(aa))
                cc += 1


        plt.clf()
        plt.figure(figsize=(5,4))
        plt.plot(labelstr[:],globaldistri[0:3],'.-',label='Full model [dist]',color='C0')
        plt.plot(labelstr[:],globalmaxprob[0:3],'.--',label='Full model [max]',color='C0',alpha=0.5)
        plt.plot(labelstr[:],globaldistri[3:6],'.-',label='Compressed [dist]',color='C1')
        plt.plot(labelstr[:],globalmaxprob[3:6],'.--',label='Compressed [max]',color='C1',alpha=0.5)
        plt.plot(labelstr[:],globaldistri[6:9],'.-',label='Resampled [dist]',color='C2')
        plt.plot(labelstr[:],globalmaxprob[6:9],'.--',label='Resampled [max]',color='C2',alpha=0.5)

        plt.locator_params(axis='y', nbins=4)
        plt.xscale('log')
        plt.xlabel('Dataset size - N')
        plt.ylabel(r'Average error - $\sigma$')
        plt.axhline(1e-2,color='k',ls='--')
        plt.minorticks_on()
        plt.legend()
        plt.savefig(self.args.directory+'plots_curve_final_network.pdf')


if __name__ == "__main__":

    myflow = bayes_inversion()
    # myflow.create_database(spectral_range=0, tauvalues = 9, noise=1e-2, size=2e6)
    # myflow.test_uncertainty2(name_posterior = '', tauvalues = 9,spirange=[5],gotostic = False)

    # myflow.test_pp_plot(name_posterior = 'posterior_15_10_64e_t9_1e-2g_1e6',tauvalues = 9,spectral_range=5)

    # myflow.test_plots_paper(name_posterior = 'posterior_15_10_64e_t9_1e-2g_1e6',tauvalues = 9,spirange=[0,5],testindex=11387,gotostic = False)
    # myflow.test_plots_paper(name_posterior = 'posterior_15_10_64e_t9_1e-2g_1e6',tauvalues = 9,spirange=[0,5],testindex=17954,gotostic = False)
    # myflow.test_plots_paper(name_posterior = 'posterior_15_10_64e_t9_1e-2g_1e6',tauvalues = 9,spirange=[0,5],testindex=8160,gotostic = False)

    # myflow.test_plots_paper(name_posterior = 'qposterior_encoding_15_10_64e_t9_1e-2g_1e6',tauvalues = 9,spirange=[5],testindex=11387,gotostic = False)
    # myflow.test_plots_paper(name_posterior = 'qposterior_encoding_15_10_64e_t9_1e-2g_1e6',tauvalues = 9,spirange=[5],testindex=17954,gotostic = False)
    # myflow.test_plots_paper(name_posterior = 'qposterior_encoding_15_10_64e_t9_1e-2g_1e6',tauvalues = 9,spirange=[5],testindex=8160,gotostic = False)

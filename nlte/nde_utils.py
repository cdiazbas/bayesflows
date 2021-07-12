# -*- coding: utf-8 -*-
__author__ = 'carlos.diaz'
# Utils

import torch
import numpy as np

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def closest(list1, value):
    return np.argmin(np.abs(list1-value))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def closest_array(list1, list2):
    index = []
    for value in list2:
        index.append(np.argmin(np.abs(list1-value)))
    return np.array(index)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sorting(list1, list2):
    list1, list2 = zip(*sorted(zip(list1, list2)))
    return np.array(list1), np.array(list2)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def write_summary(text,name="summary.txt"):
    """Write a text (dictionary) in a file"""
    if isinstance(text, dict):
        text = str(text)
    f = open(name, "a")
    f.write(text)
    f.write('\n')
    f.close()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def create_dir(namedir):
    """Creates a directory"""
    if not os.path.exists(namedir):
        os.makedirs(namedir)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_params(modelclass, verbose=True):
    """Outputs the number of parameters to optimize"""
    pytorch_total_params = sum(p.numel() for p in modelclass.parameters()) # numel returns the total number of elements
    pytorch_total_params_grad = sum(p.numel() for p in modelclass.parameters() if p.requires_grad)
    # print('[INFO] Total params (fixed and free)   :', pytorch_total_params)
    if verbose is True: print('[INFO] Total free parameters:', pytorch_total_params_grad)
    return pytorch_total_params_grad


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class basicDataset(torch.utils.data.Dataset):
  """Characterizes a basic dataset for PyTorch"""
  def __init__(self, observations, modelparameters, noise=1e-9, xnoise=1e-9, amplitude=1.0):
        """Initialization"""
        self.observations = observations
        self.modelparameters = modelparameters
        self.noise = noise
        self.xnoise = xnoise
        self.amplitude = amplitude

  def __len__(self):
        """Denotes the total number of samples"""
        return self.modelparameters.shape[0]

  def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.modelparameters[index,:]
        y = self.observations[index,:]
        ynoise = 0.0
        xnoise = 0.0

        # import time
        # mseed = int(time.time())
        # fix_seed(mseed)

        if self.noise > 1e-6:
            ynoise = np.random.normal(0.,self.noise,size=(y.shape)).astype(np.float32)
        if self.xnoise > 1e-6:
            xnoise = self.amplitude*np.random.normal(0.,self.xnoise,size=(x.shape)).astype(np.float32)
        
        # print(self.amplitude)        
        # print(self.xnoise)        
        # print(x.shape)
        # print(xnoise.shape)
        # print('===')
        # print(x)
        # print(xnoise)
        # print(y)
        # print(ynoise)
        # print(mseed)
        # preliminar =  x+xnoise
        # booli = preliminar<0
        # preliminar[18:][booli[18:]] = np.abs(preliminar[18:][booli[18:]])
        # # print(booli[18:])
        # xnoise[18:][booli[18:]] = np.abs(xnoise[18:][booli[18:]])
        
        return x+xnoise, y+ynoise
        # return preliminar, y+ynoise


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def basicLoader(observations, modelparameters, noise=0.0, xnoise=1e-9, amplitude=1.0, **kwargs):
    """Creates a basic Pytorch loader"""
    batch_size=120
    shuffle=True
    if 'batch_size' in kwargs.keys(): batch_size=kwargs.pop('batch_size')
    if 'shuffle' in kwargs.keys(): shuffle=kwargs.pop('shuffle')
    return torch.utils.data.DataLoader(basicDataset(observations, modelparameters,noise=noise,xnoise=xnoise, amplitude=amplitude), batch_size=batch_size,shuffle=shuffle, **kwargs)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def fix_seed(my_seed):
    """Fix the randomness for reproducibility"""
    import numpy as np
    import torch
    import random
    torch.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def my_subplot2grid(fig, shape, loc, rowspan=1, colspan=1,topi=None,lefti=None, **kwargs):
    from matplotlib.gridspec import GridSpec
    from brokenaxes import brokenaxes

    s1, s2 = shape
    subplotspec = GridSpec(s1, s2,bottom=topi,left=lefti).new_subplotspec(loc,
                                                   rowspan=rowspan,
                                                   colspan=colspan)
    a = fig.add_subplot(subplotspec, **kwargs)
    bbox = a.bbox
    byebye = []
    for other in fig.axes:
        if other==a: continue
        if bbox.fully_overlaps(other.bbox):
            byebye.append(other)
    for ax in byebye: plt.delaxes(ax)
    return a


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def importance_reweighting(samples, logprob, new_logprob,linear=False,truncated=False):
    # https://readthedocs.org/projects/dynesty/downloads/pdf/latest/
    # From: http://greg-ashton.physics.monash.edu/importance-reweighting-example.html
    # https://notebook.community/joshspeagle/dynesty/demos/Examples%20--%20Importance%20Reweighting
    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata To do to fix few samples.

    plot_option = False

    # samples = samples[~np.isnan(logprob)]
    # logprob = logprob[~np.isnan(logprob)]
    # new_logprob = new_logprob[~np.isnan(logprob)]

    if linear ==True:
        # Assuming good correlation we avoid problems with few samples
        coef = np.polyfit(logprob,new_logprob,1)
        poly1d_fn = np.poly1d(coef)
        print('Linear regression:',coef)
        new_logprob = poly1d_fn(logprob)

    if plot_option:
        # +++++++++++++++++++++++++++++++++++++++++++++++
        import matplotlib.pyplot as plt
        coef = np.polyfit(logprob,new_logprob,1)
        poly1d_fn = np.poly1d(coef)
        print('Linear regression:',coef)
        plt.figure()
        plt.plot(logprob,new_logprob,'.',alpha=0.5)
        plt.plot(logprob,poly1d_fn(logprob),'--')
        plt.text(logprob.min(),new_logprob.max(), 'Linear regression: {0:1.1f}, {1:1.1f}'.format(coef[0],coef[1]))
        plt.xlabel('Logprob_flow');plt.ylabel('Logprob_likelihood')
        cte = new_logprob.max() - logprob.max()
        plt.ylim(logprob.min()+ cte, new_logprob.max()*1.05)
        plt.savefig('logtest.png')
        # +++++++++++++++++++++++++++++++++++++++++++++++

    weights = new_logprob - logprob 
    weights = weights - np.mean(weights) # To avoid overflow values
    # The normalization cte does not matter, only the ratio

    # Truncated:
    if truncated is True:
        trc = np.percentile(weights, 99.7)
        print('np.percentile(weights, 90)',trc)
        weights[weights>trc] = trc

    p = np.exp(weights)
    p /= np.sum(p)

    # print('np.max(p)',np.max(p))
    # print('np.max(weights)',np.max(weights))
    # print('sorted(weights)',sorted(weights))
    # print('sorted(p)',sorted(p))

    samples = samples[~np.isnan(p)]
    new_logprob = new_logprob[~np.isnan(p)]

    if plot_option:
        # +++++++++++++++++++++++++++++++++++++++++++++++
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(p,bins=100)
        plt.savefig('loghist.png')
        # +++++++++++++++++++++++++++++++++++++++++++++++

        # +++++++++++++++++++++++++++++++++++++++++++++++
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(weights,bins=100)
        plt.savefig('loghist2.png')
        # +++++++++++++++++++++++++++++++++++++++++++++++

    newsize = samples.shape[0]
    choice = np.random.choice(np.arange(samples.shape[0]), size=newsize, p=p)
    return samples[choice,:].astype(np.float32), new_logprob[choice].astype(np.float32)


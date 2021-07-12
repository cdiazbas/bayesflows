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
  def __init__(self, observations, modelparameters, noise=0.0):
        """Initialization"""
        self.observations = observations
        self.modelparameters = modelparameters
        self.noise = noise

  def __len__(self):
        """Denotes the total number of samples"""
        return self.modelparameters.shape[0]

  def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.modelparameters[index,:]
        y = self.observations[index,:]
        ynoise = 0.0
        if self.noise > 1e-8:
            ynoise = np.random.normal(0.,self.noise,size=(y.shape)).astype(np.float32)
        return x, y+ynoise


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def basicLoader(observations, modelparameters, noise=0.0, **kwargs):
    """Creates a basic Pytorch loader"""
    batch_size=120
    shuffle=True
    if 'batch_size' in kwargs.keys(): batch_size=kwargs.pop('batch_size')
    if 'shuffle' in kwargs.keys(): shuffle=kwargs.pop('shuffle')
    return torch.utils.data.DataLoader(basicDataset(observations, modelparameters,noise=noise), batch_size=batch_size,shuffle=shuffle, **kwargs)


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
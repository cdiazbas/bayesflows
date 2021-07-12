""" Normalizing flows """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows import distributions, flows, transforms, utils
from nflows.nn import nets

# Based on https://github.com/stephengreen/lfi-gw/blob/master/lfigw/nde_flows.py


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def create_linear_transform(input_size):
    """Creates a linear transform by concatenating a permutation and LU factorization
    """
    linear_transform = transforms.CompositeTransform([
        transforms.RandomPermutation(features=input_size),
        transforms.LULinear(input_size, identity_init=True) ])
    return linear_transform


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def create_base_transform(iflow, input_size, hidden_size, context_size, num_blocks=1, transformtype="affine-autoregressive", activation=F.elu,num_bins=8):
    """Creates the base transform
    """

    # Code is adapted from https://github.com/bayesiains/nsf
    if transformtype == "rq-coupling":
        # Piecewise Rational Quadratic Coupling
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(input_size, even=(iflow % 2 == 0)),
            transform_net_create_fn=(lambda in_features, out_features: nets.ResidualNet(in_features=in_features, out_features=out_features,
                                        hidden_features=hidden_size, context_features=context_size,num_blocks=num_blocks,activation=activation,
                                        )),
            num_bins=num_bins,
            tails='linear',
            tail_bound=5,
            apply_unconditional_transform=False
        )


    elif transformtype == "affine-autoregressive":
        return transforms.MaskedAffineAutoregressiveTransform(
            features=input_size,
            hidden_features=hidden_size,
            context_features=context_size,
            num_blocks=num_blocks,
            activation=activation,
        )

    elif transformtype == "rq-autoregressive":
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=input_size,
            hidden_features=hidden_size,
            context_features=context_size,
            num_bins=num_bins,
            tails='linear',
            tail_bound=5,
            num_blocks=num_blocks,
            activation=activation,
        )

    else:
        raise ValueError


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class mininet(nn.Module):
    def __init__(self, n_input, n_output):
        super(mininet, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.L1 = nn.Linear(self.n_input, self.n_output) 
    def forward(self, x):
        out = self.L1(x)
        return out


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def create_flow(y_size, x_size, num_flows=5, mhidden_features=32, num_blocks=1, transformtype="rq-coupling",embedding_net=None, base_dist_net=None, conditional=False,context_encoder=None,num_bins=8):
    """Creates the set of transformations from the base distribution
    """
    
    if base_dist_net is None:
        base_dist = distributions.StandardNormal((y_size,))
        if conditional is True:
            if context_encoder is None:
                context_encoder_net = mininet(n_input=x_size,n_output=y_size*2)
            else:
                context_encoder_net = context_encoder
            base_dist = distributions.ConditionalDiagonalNormal(shape=(y_size,), context_encoder=context_encoder_net)
    else:
        base_dist = base_dist_net


    transformsi = []
    for iflow in range(num_flows):
        transformsi.append(create_linear_transform(y_size))
        transformsi.append(create_base_transform(iflow, y_size, mhidden_features, x_size,
        num_blocks=num_blocks, transformtype=transformtype,num_bins=num_bins))
    transformsi.append(create_linear_transform(y_size))
    transformflow = transforms.CompositeTransform(transformsi)

    if embedding_net is None:
        embedding = nn.Identity()
    else:
        embedding = embedding_net


    modelflow = Flow(transformflow, base_dist, embedding)

    return modelflow


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class NFLOW(nn.Module):
    """Conditional Normalizing Flows
    """
    # transformtype : [ "rq-coupling", 'affine-autoregressive', "rq-autoregressive" ]
    def __init__(self, y_size, x_size, num_flows=5, mhidden_features=32, num_blocks=1, train_loader=None, 
        embedding_net=None, transformtype="rq-coupling", device="cpu", base_dist_net=None,conditional=False,
        context_encoder=None,nbins=8):

        super(NFLOW, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.modelflow = create_flow(self.y_size, self.x_size, num_flows=num_flows, mhidden_features=mhidden_features,
            num_blocks=num_blocks, embedding_net=embedding_net, transformtype=transformtype,base_dist_net=base_dist_net,
            conditional=conditional,context_encoder=context_encoder,num_bins=nbins)

        if train_loader is not None:
            print('[INFO] Normalizing to training set')
            self.x_std = train_loader.dataset.observations.std((0,1))
            self.x_mean = train_loader.dataset.observations.mean((0,1))
            self.y_std = train_loader.dataset.modelparameters.std((0))
            self.y_mean = train_loader.dataset.modelparameters.mean((0))
        else:
            self.y_std = 1.0
            self.y_mean = 0.0
            self.x_std = 1.0
            self.x_mean = 0.0

        
        # -------- CUDA ----------------
        if device == "cuda":
            self.y_std = torch.from_numpy(self.y_std).to(device)
            self.y_mean = torch.from_numpy(self.y_mean).to(device)
            self.x_mean = torch.from_numpy(np.array(self.x_mean)).to(device)
            self.x_std = torch.from_numpy(np.array(self.x_std)).to(device)
        # -------- CUDA ----------------


    def forward(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        loss = -self.modelflow.log_prob(inputs=y, context=x).mean()
        return loss

    def log_prob(self,inputs,context):
        return self.modelflow.log_prob(inputs=inputs, context=context)

    def sample(self, x):
        # Draw samples from the posterior
        x = (x-self.x_mean)/self.x_std
        return self.modelflow.sample(1,context=x)[:,0,:].detach() *self.y_std + self.y_mean

    # def inverse(self, y):
    #     y = y/self.y_std
    #     return self.modelflow.transform_to_noise(y)*self.x_std

    def obtain_samples(self, x, nsamples, batch_size=512):
        # Draw samples from the posterior
        x = (x-self.x_mean)/self.x_std
        # print(self.x_std,self.x_mean,self.y_std,self.y_mean)

        with torch.no_grad():
            self.modelflow.eval()
            
            if torch.is_tensor(x) is False:
                x = torch.from_numpy(x)
            # x = torch.from_numpy(x).unsqueeze(0)#.to(device)

            num_batches = nsamples // batch_size
            num_leftover = nsamples % batch_size

            samples = [self.modelflow.sample(batch_size, x) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self.modelflow.sample(num_leftover, x))

            return torch.cat(samples, dim=1)[0]*self.y_std + self.y_mean


    def sample_and_log_prob(self, x, nsamples, batch_size=512):
        x = (x-self.x_mean)/self.x_std
        
        with torch.no_grad():
            self.modelflow.eval()

            # x = torch.from_numpy(x).unsqueeze(0)#.to(device)
            if torch.is_tensor(x) is False:
                x = torch.from_numpy(x)

            num_batches = nsamples // batch_size
            num_leftover = nsamples % batch_size

            samples = [self.modelflow.sample(batch_size, x) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self.modelflow.sample(num_leftover, x))

        total_samples = torch.cat(samples, dim=1)[0]
        log_prob = self.modelflow.log_prob(total_samples[:,:], x.repeat(total_samples.shape[0],x.shape[0]))
        return total_samples.data.cpu().numpy()*self.y_std + self.y_mean, log_prob.data.cpu().numpy()


    def sample_and_log_prob_noise(self, x, nsamples, batch_size=512, extranoise=0.0):
        x = (x-self.x_mean)/self.x_std
        
        with torch.no_grad():
            self.modelflow.eval()

            x = torch.from_numpy(x)#.unsqueeze(0)#.to(device)

            num_batches = nsamples // batch_size
            num_leftover = nsamples % batch_size

            # print(x.shape,batch_size,num_batches)
            # # + torch.randn(4) 

            # from ipdb import set_trace as stop
            # stop()

            samples = [self.modelflow.sample(batch_size, x) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self.modelflow.sample(num_leftover, x))

        total_samples = torch.cat(samples, dim=1)[0]

        if extranoise > 1e-8:
            x2 = x.repeat(nsamples,1)
            x2 = torch.randn(x2.shape)*extranoise/self.x_std +x2
            total_samples = self.modelflow.sample(1, x2)[:,0,:]
            # print(extranoise/self.x_std)

        log_prob = self.modelflow.log_prob(total_samples[:,:], x.repeat(total_samples.shape[0],x.shape[0]))
        return total_samples.data.cpu().numpy()*self.y_std + self.y_mean, log_prob.data.cpu().numpy()


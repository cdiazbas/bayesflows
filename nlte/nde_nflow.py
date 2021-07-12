""" Normalizing flows """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows import distributions, flows, transforms, utils
from nflows.nn import nets
import nde_utils
from ipdb import set_trace as stop
# ipdb> self.model.lstflows[0]._transform._transforms[0]._transforms[0]._permutation

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def create_linear_transform(input_size):
    """Creates a linear transform by concatenating a permutation and LU factorization
    """
    linear_transform = transforms.CompositeTransform([
        transforms.RandomPermutation(features=input_size),
        transforms.LULinear(input_size, identity_init=True) ])
    return linear_transform


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def create_base_transform(iflow, input_size, hidden_size, context_size, num_blocks=1, transformtype="rq-autoregressive", activation=F.elu,num_bins=8):
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
        transformsi.append(create_base_transform(iflow, y_size, mhidden_features, x_size,num_blocks=num_blocks, transformtype=transformtype,num_bins=num_bins))
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
        context_encoder=None,output_decoder=None,input_encoder=None,num_bins=8):

        super(NFLOW, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.output_decoder = output_decoder
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.modelflow = create_flow(self.y_size, self.x_size, num_flows=num_flows, mhidden_features=mhidden_features,
            num_blocks=num_blocks, embedding_net=embedding_net, transformtype=transformtype,base_dist_net=base_dist_net,
            conditional=conditional,context_encoder=context_encoder,num_bins=num_bins)

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

        if output_decoder is not None:
            print('[INFO] Using a pretrained output decoder')
            self.output_decoder = torch.load(output_decoder)
            for parami in self.output_decoder.parameters():
                parami.requires_grad = False

        if input_encoder is not None:
            print('[INFO] Using a pretrained input encoder')
            self.input_encoder = torch.load(input_encoder)
            for parami in self.input_encoder.parameters():
                parami.requires_grad = False
            # self.x_std = self.input_encoder.x_std
            # self.x_mean = self.input_encoder.x_mean
            self.input_encoder.x_std = 1.0
            self.input_encoder.x_mean = 0.0

    def forward(self, y, x, weight=False):
        if y.is_cuda is True: self.y_mean,self.y_std = self.y_mean.cuda(),self.y_std.cuda()

        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        try: 
            if self.input_encoder is not None: x = self.input_encoder.sample_encoder(x)
        except: pass
        # loss = -self.modelflow.log_prob(inputs=y, context=x).mean()
        loss = self.modelflow.log_prob(inputs=y, context=x)
        return -loss.mean()
        
        # if weight is False:
        #     return -loss.mean()
        # else:
        #     w = torch.exp(loss)
        #     w = w/w.mean()
        #     return -loss.mean(), -torch.mean(loss*w)


    def log_prob(self,inputs,context):
        return self.modelflow.log_prob(inputs=inputs, context=context)

    def sample(self, x):
        # Draw samples from the posterior
        x = (x-self.x_mean)/self.x_std
        try: 
            if self.input_encoder is not None: x = self.input_encoder.sample_encoder(x)
        except: pass
        return self.modelflow.sample(1,context=x)[:,0,:].detach() *self.y_std + self.y_mean

    # def inverse(self, y):
    #     y = y/self.y_std
    #     return self.modelflow.transform_to_noise(y)*self.x_std

    def obtain_samples(self, x, nsamples, batch_size=512):
        # Draw samples from the posterior
        x = (x-self.x_mean)/self.x_std
        try: 
            if self.input_encoder is not None: x = self.input_encoder.sample_encoder(x)
        except: pass

        # print(self.x_std,self.x_mean,self.y_std,self.y_mean)

        with torch.no_grad():
            self.modelflow.eval()

            if torch.is_tensor(x) is False: x = torch.from_numpy(x).unsqueeze(0)#.to(device)

            try: x = x.cuda()
            except: pass

            num_batches = nsamples // batch_size
            num_leftover = nsamples % batch_size

            samples = [self.modelflow.sample(batch_size, x) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self.modelflow.sample(num_leftover, x))
        
            try:
                if self.output_decoder is not None:
                    # print(self.y_std,self.y_mean,torch.cat(samples, dim=1))
                    vector = torch.cat(samples, dim=1)[0]*self.y_std + self.y_mean
                    return self.output_decoder.sample_decoder(vector)
                else:
                    return torch.cat(samples, dim=1)[0]*self.y_std + self.y_mean
            except:
                return torch.cat(samples, dim=1)[0]*self.y_std + self.y_mean


    def sample_and_log_prob(self, x, nsamples, batch_size=512, extranoise=0.0):
        x = (x-self.x_mean)/self.x_std
        try:
            if self.input_encoder is not None: x = self.input_encoder.sample_encoder(x)
        except: pass

        
        with torch.no_grad():
            self.modelflow.eval()

            if torch.is_tensor(x) is False: x = torch.from_numpy(x).unsqueeze(0)#.to(device)
            
            num_batches = nsamples // batch_size
            num_leftover = nsamples % batch_size

            # print(x.shape,batch_size,num_batches)
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
        # return total_samples.data.cpu().numpy()*self.y_std + self.y_mean, log_prob.data.cpu().numpy()

        try:
            if self.output_decoder is not None:
                return self.output_decoder.sample_decoder(total_samples.data.cpu()*self.y_std + self.y_mean).numpy(), log_prob.data.cpu().numpy()
            else:
                return total_samples.data.cpu().numpy()*self.y_std + self.y_mean, log_prob.data.cpu().numpy()
        except:
            return total_samples.data.cpu().numpy()*self.y_std + self.y_mean, log_prob.data.cpu().numpy()




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class MFLOW(nn.Module):
    """Multi Conditional Normalizing Flows
    """
    # transformtype : [ "rq-coupling", 'affine-autoregressive', "rq-autoregressive" ]
    def __init__(self, y_size, x_size, num_flows=5, mhidden_features=32, num_blocks=1, train_loader=None, 
        embedding_net=None, transformtype="rq-coupling", device="cpu", base_dist_net=None,conditional=False,
        context_encoder=None, composition=None ):

        super(MFLOW, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if composition is None:
            self.composition = range(0,len(y_size))
        else:
            # self.composition = [ [0,1,2,3,4,5,6,7,8], [9,10,11,12,13,14,15,16,17], [18,19,20,21,22,23,24,5,26]]
            # self.composition = [ [0,9,18], [1,10,19], [2,11,20], [3,12,21], [4,13,22], [5,14,23], [6,15,24], [7,16,25], [8,17,26]]
            self.composition = composition
        self.flat_list = [item for sublist in self.composition for item in sublist]
        self.simple_list = range(0,len(self.flat_list))
        dum, self.final_lst = nde_utils.sorting(self.flat_list,self.simple_list)
        # stop()

        self.lstflows = []
        for m in range(len(self.composition)):
            m_size = len(self.composition[m])
            self.lstflows.append( create_flow(m_size, self.x_size, num_flows=num_flows, mhidden_features=mhidden_features,
            num_blocks=num_blocks, embedding_net=embedding_net, transformtype=transformtype,base_dist_net=base_dist_net,
            conditional=conditional,context_encoder=context_encoder) )

        self.lstflows = nn.ModuleList(self.lstflows)

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
        loss = self.log_prob_multi(inputs=y, context=x)
        return loss

    def log_prob_multi(self,inputs,context):
        loss = 0
        for m in range(len(self.composition)):
            loss += -self.lstflows[m].log_prob(inputs[:,self.composition[m]], context).mean()
        return loss

    def sample_multi(self, nsample, context):
        output = []
        for m in range(len(self.composition)):
            output.append( self.lstflows[m].sample(nsample,context) )

        # stop()
        return torch.cat(output, dim=2)[:,:,self.final_lst]


    def sample(self, x):
        # Draw samples from the posterior
        x = (x-self.x_mean)/self.x_std
    
        if torch.is_tensor(x) is False:
            x = torch.from_numpy(x)

        return self.sample_multi(1,context=x)[:,0,:].detach() *self.y_std + self.y_mean


    def obtain_samples(self, x, nsamples, batch_size=512):

        x = (x-self.x_mean)/self.x_std

        if torch.is_tensor(x) is False:
            # x = torch.from_numpy(x)
            x = torch.from_numpy(x).unsqueeze(0)#.to(device)


        with torch.no_grad():
            # self.modelflow.eval()

            num_batches = nsamples // batch_size
            num_leftover = nsamples % batch_size

            samples = [self.sample_multi(batch_size, x) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self.sample_multi(num_leftover, x))

            return torch.cat(samples, dim=1)[0]*self.y_std + self.y_mean


    def sample_and_log_prob(self, x, nsamples, batch_size=512, extranoise=0.0):
        x = (x-self.x_mean)/self.x_std
        
        if torch.is_tensor(x) is False:
            x = torch.from_numpy(x).unsqueeze(0)#.to(device)

        with torch.no_grad():
            # self.modelflow.eval()

            num_batches = nsamples // batch_size
            num_leftover = nsamples % batch_size

            # from ipdb import set_trace as stop
            # stop()

            samples = [self.sample_multi(batch_size, x) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self.sample_multi(num_leftover, x))

        total_samples = torch.cat(samples, dim=1)[0]

        if extranoise > 1e-8:
            x2 = x.repeat(nsamples,1)
            x2 = torch.randn(x2.shape)*extranoise/self.x_std +x2
            total_samples = self.sample_multi(1, x2)[:,0,:]
            # print(extranoise/self.x_std)

        log_prob = self.log_prob_multi(total_samples[:,:], x.repeat(total_samples.shape[0],x.shape[0]))
        return total_samples.data.cpu().numpy()*self.y_std + self.y_mean, log_prob.data.cpu().numpy()



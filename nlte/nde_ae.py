import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class create_Coder(nn.Module):
    """Creates a encoder or a decoder according to input parameters
    """
    def __init__(self, input_size, output_size, hidden_size=[64,64], activation=nn.ELU()):
        super(create_Coder, self).__init__()
        self.activation = activation

        # Input layers
        coder_net_list = []
        coder_net_list.append( nn.Linear(input_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            coder_net_list.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.coder_net_list = nn.ModuleList(coder_net_list)

        # Output layers
        self.output_mu = nn.Linear(hidden_size[-1], output_size)        

    def forward(self, xx):
        
        # Pass through input layers
        for ii, ilayer in enumerate(self.coder_net_list):
            xx = self.activation(ilayer(xx))

        # Pass through output layers
        mu = self.output_mu(xx)
        return mu




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class AE(nn.Module):
    """ Autoencoder (AE) [Vanilla]
    """
    def __init__(self, x_size, latent_size, hidden_size = [64, 64, 64, 64, 64], train_loader=None):
        super(AE, self).__init__()
        self.x_size = x_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder_ = create_Coder(self.x_size, self.latent_size, hidden_size=self.hidden_size)
        self.decoder_ = create_Coder(self.latent_size, self.x_size, hidden_size=self.hidden_size)
        
        # print('train_loader.dataset.observations.shape',train_loader.dataset.observations.shape)
        if train_loader is not None:
            self.x_std = train_loader.dataset.observations.std((0,1))
        else:
            self.x_std = 1.0

    def encode(self, x): # x
        # Encoder
        mu_E  = self.encoder_(x)
        return mu_E

    def decode(self, z):
        # Decoder
        mu_D  = self.decoder_(z)
        return mu_D 

    def forward(self, x):
        x = x/self.x_std
        mu_E = self.encode(x.view(-1, self.x_size))
        mu_D = self.decode(mu_E)
        loss  = self.loss_function(x, mu_D)
        return loss

    def loss_function(self, x, output):
        # Reconstruction loss
        rec = torch.mean( (output-x)**2.)
        return rec

    def sample(self, x, latent=False):
        if torch.is_tensor(x) is False:
            # x = torch.from_numpy(x)
            x = torch.from_numpy(x).unsqueeze(0)#.to(device)

        # Forward pass
        x = x/self.x_std
        mu_E = self.encode(x.view(-1, self.x_size))
        mu_D = self.decode(mu_E)

        if latent is True:
            return mu_D*self.x_std, mu_E
        else:
            return mu_D*self.x_std






# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""
    def __init__(self, in_features, out_features, hidden_features, context_features=None, num_blocks=2, activation=F.elu, dropout_probability=0.0, use_batch_norm=False,
        train_loader=None,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mu = nn.Linear(hidden_features, out_features)
        self.output_logvar = nn.Linear(hidden_features, out_features)
        
        if train_loader is not None:
            print('[INFO] Normalizing to training set')
            self.x_std = train_loader.dataset.modelparameters.std((0))
            self.x_mean = train_loader.dataset.modelparameters.mean((0))
            self.y_std = train_loader.dataset.observations.std((0))
            self.y_mean = train_loader.dataset.observations.mean((0))
        else:
            self.y_std = 1.0
            self.y_mean = 0.0
            self.x_std = 1.0
            self.x_mean = 0.0

        self.l1loss = nn.L1Loss()

    def forward_mu(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        
        # Pass through output layers
        mu = self.output_mu(temps)
        logvar = self.output_logvar(temps)
        return mu

    def forwardX(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        
        # Pass through output layers
        mu = self.output_mu(temps)
        logvar = self.output_logvar(temps)

        dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar.mul(0.5))),1)
        return dist

    def forward(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        dist = self.forwardX(x)
        return self.loss_function(y, dist)

    def loss_function(self, y, dist):
        # Reconstruction loss
        neg_logp_total = dist.log_prob(y)
        return -torch.mean(neg_logp_total)

    def sample(self, x):
        # Forward pass
        x = (x-self.x_mean)/self.x_std
        dist = self.forwardX(x)
        # Sampling
        return dist.sample()*self.y_std + self.y_mean

    def obtain_samples(self, x, nsamples):
        # Forward pass
        x = (x-self.x_mean)/self.x_std
        x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        dist = self.forwardX(x)
        return dist.sample((nsamples,))*self.y_std + self.y_mean

    def evaluate(self, y, x):
        # y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        # x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        # return self.forward_mu(x).detach().numpy()*self.y_std + self.y_mean
        return self.forward_mu(x)*self.y_std + self.y_mean

    def forward_mse(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        mu = self.forward_mu(x)
        return self.loss_function_mse(y, mu)

    def loss_function_mse(self, y, dist):
        # print(y.shape,dist.shape)
        # return +torch.mean(torch.abs(y - dist))
        return self.l1loss(y,dist)



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from torch.nn import init

class RAE(nn.Module):
    """ Residual Autoencoder (RAE)
    """
    def __init__(self, x_size, latent_size, hidden_size = 64,num_blocks=4, train_loader=None):
        super(RAE, self).__init__()

        self.x_size = x_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder_ = ResidualNet(self.x_size, self.latent_size, hidden_features=self.hidden_size,num_blocks=num_blocks)
        self.decoder_ = ResidualNet(self.latent_size, self.x_size, hidden_features=self.hidden_size,num_blocks=num_blocks)
        
        # print('train_loader.dataset.observations.shape',train_loader.dataset.observations.shape)
        if train_loader is not None:
            self.x_std = train_loader.dataset.modelparameters.std((0))
            self.x_mean = train_loader.dataset.modelparameters.mean((0))

        else:
            self.x_std = 1.0
            self.x_mean = 0.0
        

    def encode(self, x): # x
        # Encoder
        mu_E  = self.encoder_.evaluate(None,x)
        return mu_E

    def decode(self, z):
        # Decoder
        mu_D  = self.decoder_.evaluate(None,z)
        return mu_D 

    def forward(self, x, ww=None):
        x = (x-self.x_mean)/self.x_std
        mu_E = self.encode(x.view(-1, self.x_size))
        mu_D = self.decode(mu_E)
        loss  = self.loss_function(x, mu_D, ww=ww)
        return loss

    def loss_function(self, x, output, ww=None):
        # Reconstruction loss

        if ww is None:
            ww = 1.0
        rec = torch.mean( ww*(output-x)**2.) #
        return rec

    def sample(self, x, latent=False):
        if torch.is_tensor(x) is False:
            # x = torch.from_numpy(x)
            x = torch.from_numpy(x).unsqueeze(0)#.to(device)

        # Forward pass
        x = (x-self.x_mean)/self.x_std
        mu_E = self.encode(x.view(-1, self.x_size))
        mu_D = self.decode(mu_E)

        if latent is True:
            return mu_D*self.x_std + self.x_mean, mu_E
        else:
            return mu_D.detach()*self.x_std + self.x_mean


    def sample_encoder(self, x):
        # print('[XX] = ',x.shape)
        if torch.is_tensor(x) is False:
            # x = torch.from_numpy(x)
            x = torch.from_numpy(x).unsqueeze(0)#.to(device)

        x = (x-self.x_mean)/self.x_std
        # print('[XX] = ',self.x_mean,self.x_std,self.x_size)
        mu_E = self.encode(x.view(-1, self.x_size))
        return mu_E.detach()
    
    def sample_decoder(self, mu_E):
        mu_D = self.decode(mu_E)
        if mu_D.is_cuda is True: #self.x_std,self.x_mean = torch.from_numpy(self.x_std).cuda(),torch.from_numpy(self.x_mean).cuda()
            return mu_D.detach()*torch.from_numpy(self.x_std).cuda() + torch.from_numpy(self.x_mean).cuda()
        else:
            return mu_D.detach()*self.x_std + self.x_mean


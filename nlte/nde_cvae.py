import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def _strictly_tril_size(n):
    """Unique elements outside the diagonal
    """
    return n * (n-1) // 2


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class create_Coder(nn.Module):
    """Creates a encoder or a decoder according to input parameters
    """
    def __init__(self, input_size, output_size, hidden_size=[64,64], activation=nn.ELU(), full_cov=True, methodfullcov="scale_tril"):
        super(create_Coder, self).__init__()
        self.activation = activation
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.epsrho = 1e-2
        self.full_cov = full_cov
        self.methodfullcov = methodfullcov

        # Input layers
        coder_net_list = []
        coder_net_list.append( nn.Linear(input_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            coder_net_list.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.coder_net_list = nn.ModuleList(coder_net_list)

        # Output layers
        self.output_mu = nn.Linear(hidden_size[-1], output_size)
        self.output_logvar = nn.Linear(hidden_size[-1], output_size)
        
        if self.full_cov is True:
            self.output_chol_net = nn.Linear(hidden_size[-1], _strictly_tril_size(output_size))
            self.lt_indices = torch.tril_indices(output_size, output_size, -1)
    
    def forward(self, xx):
        
        # Pass through input layers
        for ii, ilayer in enumerate(self.coder_net_list):
            xx = self.activation(ilayer(xx))

        # Pass through output layers
        mu = self.output_mu(xx)
        logvar = -self.elu(self.output_logvar(xx))

        if self.full_cov is False:
            dist = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(logvar.mul(0.5))),1)

            return mu, logvar, dist
        else:

            if self.methodfullcov == "scale_tril":

                rho = self.output_chol_net(xx)

                diagA = torch.exp(logvar.mul(0.5))
                diagB = torch.exp(logvar)
                diag = torch.diag_embed(diagA)
                chol = torch.zeros_like(diag)
                chol[..., self.lt_indices[0], self.lt_indices[1]] = rho
                chol = chol + diag
                dist = torch.distributions.MultivariateNormal(loc=mu, scale_tril=chol)

            elif self.methodfullcov == "covariance_matrix":
                rho = self.tanh(self.output_chol_net(xx))* (1.0- self.epsrho)
               
                diagA = torch.exp(logvar.mul(0.5))
                diagB = torch.exp(logvar)
                diag = torch.diag_embed(diagB)
                chol = torch.zeros_like(diag)

                for jj in range(self.lt_indices.shape[1]):
                    chol[..., self.lt_indices[0][jj], self.lt_indices[1][jj]] = rho[:,jj]*diagA[:,self.lt_indices[0][jj]]*diagA[:,self.lt_indices[1][jj]]
                    chol[..., self.lt_indices[1][jj], self.lt_indices[0][jj]] = rho[:,jj]*diagA[:,self.lt_indices[0][jj]]*diagA[:,self.lt_indices[1][jj]]
                chol = chol + diag
                # torch.cholesky(chol)
                dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=chol)
                 
            return mu, [logvar, rho], dist



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class CVAE(nn.Module):
    """Conditional Variational Autoencoder (CVAE)
    """
    def __init__(self, y_size, x_size, latent_size, full_cov=True, hidden_size = [64, 64, 64], train_loader=None, methodfullcov="scale_tril"):
        super(CVAE, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.full_cov = full_cov
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.methodfullcov = methodfullcov
        
        self.encoder1_ = create_Coder(self.x_size, self.latent_size, full_cov=False, hidden_size=self.hidden_size)
        
        self.encoder2_ = create_Coder(self.y_size + self.x_size, self.latent_size, full_cov=False, hidden_size=self.hidden_size)
        
        self.decoder_ = create_Coder(self.latent_size + self.x_size, self.y_size, full_cov=self.full_cov, 
            hidden_size=self.hidden_size,methodfullcov=self.methodfullcov)
        
        if self.full_cov is True: self.lt_indices = self.decoder_.lt_indices


        if train_loader is not None:
            self.x_std = train_loader.dataset.observations.std((0,1))
            self.y_std = train_loader.dataset.modelparameters.std((0))
        else:
            self.y_std = 1.0
            self.x_std = 1.0


    def encode1(self, x): # x
        # Conditional Prior network
        mu_E , logvar_E, dum = self.encoder1_(x)
        return mu_E, logvar_E

    def encode2(self, x, y): # y + x
        # Recognition network
        inputs = torch.cat([x, y], 1) 
        mu_K, logvar_K, dum = self.encoder2_(inputs)
        return mu_K, logvar_K

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar.mul(0.5))
        eps = torch.empty_like(std).normal_()
        eps.to(self.device)
        return eps.mul(std).add_(mu)

    def decode(self, z, x):
        # Generation network
        inputs = torch.cat([z, x], 1) # (bs, latent_size+class_size)
        mu_D, logvar_D, dist = self.decoder_(inputs)
        return mu_D, logvar_D, dist

    def forward(self, y, x, split=False):
        y = y/self.y_std
        x = x/self.x_std

        mu_E, logvar_E = self.encode1(x.view(-1, self.x_size))
        mu_K, logvar_K = self.encode2(x.view(-1, self.x_size), y)
        z = self.reparametrize(mu_K, logvar_K)
        mu_D, logvar_D, dist = self.decode(z,x)
        loss, i_rec, i_kld = self.loss_function(y, dist, mu_K, logvar_K, mu_E, logvar_E)
        
        if split == False:
            return loss
        else:
            return loss, i_rec, i_kld

    def loss_function(self, y, dist, mu_K, logvar_K, mu_E, logvar_E):
        # Reconstruction loss
        neg_logp_total = dist.log_prob(y)
        rec = -torch.mean(neg_logp_total)

        # KL divergence loss
        var_K = logvar_K.exp()
        var_E = logvar_E.exp()
        kl_div = var_K/var_E + (mu_K - mu_E)**2.0 / var_E + logvar_E - logvar_K - 1.0
        kld = torch.mean(kl_div.sum(dim=1) / 2.0)
        return rec+kld, rec, kld

    def sample(self, x):
        # Forward pass
        x = x/self.x_std
        mu_E, logvar_E = self.encode1(x.view(-1, self.x_size))
        z = self.reparametrize(mu_E, logvar_E)
        mu_D, logvar_D, dist = self.decode(z,x)
        return dist.sample()*self.y_std



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GSNN(nn.Module):
    """Gaussian Stochastic Neural network (GSNN)
    """
    # It is like CVAE but without the recognition encoder
    def __init__(self, y_size, x_size, latent_size, full_cov=True, hidden_size = [64, 64, 64], train_loader=None, methodfullcov="scale_tril"):
        super(GSNN, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.full_cov = full_cov
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.methodfullcov = methodfullcov

        self.encoder1_ = create_Coder(self.x_size, self.latent_size, full_cov=False, hidden_size=self.hidden_size)

        self.decoder_ = create_Coder(self.latent_size + self.x_size, self.y_size, full_cov=self.full_cov, 
            hidden_size=self.hidden_size,methodfullcov=self.methodfullcov)

        if self.full_cov is True: self.lt_indices = self.decoder_.lt_indices


        if train_loader is not None:
            self.x_std = train_loader.dataset.observations.std((0,1))
            self.y_std = train_loader.dataset.modelparameters.std((0))
        else:
            self.y_std = 1.0
            self.x_std = 1.0

    def encode1(self, x): # x
        # Conditional Prior network
        mu_E , logvar_E, dum = self.encoder1_(x)
        return mu_E, logvar_E

    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar.mul(0.5))
        eps = torch.empty_like(std).normal_()
        eps.to(self.device)
        return eps.mul(std).add_(mu)

    def decode(self, z, x):
        # Generation network
        inputs = torch.cat([z, x], 1) # (bs, latent_size+class_size)
        mu_D, logvar_D, dist = self.decoder_(inputs)
        return mu_D, logvar_D, dist

    def forward(self, y, x):
        y = y/self.y_std
        x = x/self.x_std

        mu_E, logvar_E = self.encode1(x.view(-1, self.x_size))
        z = self.reparametrize(mu_E, logvar_E)
        mu_D, logvar_D, dist = self.decode(z,x)
        loss, i_rec, i_kld = self.loss_function(y, dist, mu_E, logvar_E, mu_E, logvar_E)
        return loss

    def loss_function(self, y, dist, mu_K, logvar_K, mu_E, logvar_E):
        # Reconstruction loss
        neg_logp_total = dist.log_prob(y)
        rec = -torch.mean(neg_logp_total)

        # KL divergence loss = 0
        kld = torch.zeros([1], dtype=torch.int32)
        return rec+kld, rec, kld

    def sample(self, x):
        # Forward pass
        x = x/self.x_std
        mu_E, logvar_E = self.encode1(x.view(-1, self.x_size))
        z = self.reparametrize(mu_E, logvar_E)
        mu_D, logvar_D, dist = self.decode(z,x)
        # Sampling
        return dist.sample()*self.y_std



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GDNN(nn.Module):
    """Gaussian Deterministic Neural network (GDNN)
    """
    # It is like CVAE but without the recognition encoder and without the reparameterization trick
    def __init__(self, y_size, x_size, latent_size, full_cov=True, hidden_size = [64, 64, 64], train_loader=None, methodfullcov="scale_tril"):
        super(GDNN, self).__init__()
        self.y_size = y_size
        self.x_size = x_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.full_cov = full_cov
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.methodfullcov = methodfullcov


        self.decoder_ = create_Coder(self.x_size, self.y_size, full_cov=self.full_cov, hidden_size=self.hidden_size,methodfullcov=self.methodfullcov)

        if self.full_cov is True: self.lt_indices = self.decoder_.lt_indices

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


    def forward(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        mu_D, logvar_D, dist = self.decoder_(x)
        return self.loss_function(y, dist)

    def loss_function(self, y, dist):
        # Reconstruction loss
        neg_logp_total = dist.log_prob(y)
        return -torch.mean(neg_logp_total)

    def sample(self, x):
        # Forward pass
        x = (x-self.x_mean)/self.x_std
        mu_D, logvar_D, dist = self.decoder_(x)
        # Sampling
        return dist.sample()*self.y_std + self.y_mean

    def obtain_samples(self, x, nsamples):
        # Forward pass
        x = (x-self.x_mean)/self.x_std
        x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        mu_D, logvar_D, dist = self.decoder_(x)
        return dist.sample((nsamples,))*self.y_std + self.y_mean





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


class ResidualNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.elu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
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
                    zero_initialization=zero_initialization,
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

        # print(self.y_std)
        # print(self.x_std)
        self.l1loss = nn.L1Loss()
        self.l1loss = nn.MSELoss()

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

    def evaluate(self, x):
        # y = (y-self.y_mean)/self.y_std
        if torch.is_tensor(x) is False: x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        x = (x-self.x_mean)/self.x_std
        # x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        return self.forward_mu(x).detach().numpy()*self.y_std + self.y_mean

    def forward_mse(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        mu = self.forward_mu(x)
        return self.loss_function_mse(y, mu)

    def loss_function_mse(self, y, dist):
        # print(y.shape,dist.shape)
        # return +torch.mean(torch.abs(y - dist))
        return self.l1loss(y,dist)









class ResidualNet2(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs. Only mu!"""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.elu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
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
                    zero_initialization=zero_initialization,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mu = nn.Linear(hidden_features, out_features)
        
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

        # self.l1loss = nn.L1Loss()
        self.l1loss = nn.MSELoss()

    def forward_mu(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        
        # Pass through output layers
        mu = self.output_mu(temps)
        return mu

    def obtain_samples(self, x, nsamples):
        # Forward pass
        x = (x-self.x_mean)/self.x_std
        x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        dist = self.forward_mu(x)
        return torch.unsqueeze(dist, 0).detach().numpy()*self.y_std + self.y_mean

    def evaluate(self, x):
        # y = (y-self.y_mean)/self.y_std
        if torch.is_tensor(x) is False: x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        x = (x-self.x_mean)/self.x_std
        # x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        return self.forward_mu(x).detach().numpy()*self.y_std + self.y_mean

    def forward_mse(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        mu = self.forward_mu(x)
        return self.loss_function_mse(y, mu)

    def loss_function_mse(self, y, dist):
        # return +torch.mean(torch.abs(y - dist))
        return self.l1loss(y,dist)













class ResidualNet3(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs. 1 trainable Fourier"""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.elu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
        train_loader=None,
    ):
        super().__init__()
        fourier = 2
        in_features = int(fourier*in_features)
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
                    zero_initialization=zero_initialization,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mu = nn.Linear(hidden_features, out_features)
        
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

        # self.l1loss = nn.L1Loss()
        self.l1loss = nn.MSELoss()
        self.register_parameter(name='A0', param=torch.nn.Parameter(torch.randn(int(in_features/fourier))))
        self.register_parameter(name='w0', param=torch.nn.Parameter(torch.randn(int(in_features/fourier))))
        self.register_parameter(name='b0', param=torch.nn.Parameter(torch.randn(int(in_features/fourier))))
        # print('in_features',in_features)


    def forward_mu(self, inputs, context=None):
        # print(inputs.max((0)))
        # print(self.w0)
        

        # print(inputs.shape)
        # print(torch.sin(inputs).shape)
        # print(torch.sin(self.w0*inputs).shape)
        

        inputs = torch.cat((inputs, 3*self.A0*torch.sin(self.w0*inputs+self.b0)), dim=1)



        # print(inputs.max())
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        
        # Pass through output layers
        mu = self.output_mu(temps)
        return mu

    def obtain_samples(self, x, nsamples):
        # Forward pass
        x = (x-self.x_mean)/self.x_std
        x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        dist = self.forward_mu(x)
        return torch.unsqueeze(dist, 0).detach().numpy()*self.y_std + self.y_mean

    def evaluate(self, x):
        # y = (y-self.y_mean)/self.y_std
        # print('---------')
        # print(self.w0)
        # print(self.A0)
        # print(self.b0)
        # print('---------')

        if torch.is_tensor(x) is False: x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        x = (x-self.x_mean)/self.x_std
        # x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        return self.forward_mu(x).detach().numpy()*self.y_std + self.y_mean

    def forward_mse(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        mu = self.forward_mu(x)
        return self.loss_function_mse(y, mu)

    def loss_function_mse(self, y, dist):
        # return +torch.mean(torch.abs(y - dist))
        return self.l1loss(y,dist)













class ResidualNet4(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs. Fourier feature"""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.elu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
        train_loader=None,
        fourier = 128,
        trainble=True,
        matrix=True,
    ):
        super().__init__()
        self.matrix = matrix
        in_features_ori = int(in_features*1.0)
        in_features = int(fourier*in_features)
        if self.matrix: in_features  = int(fourier-1 + in_features_ori)  
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
                    zero_initialization=zero_initialization,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mu = nn.Linear(hidden_features, out_features)
        
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

        # self.l1loss = nn.L1Loss()
        self.l1loss = nn.MSELoss()
        # self.register_parameter(name='A0', param=torch.nn.Parameter(torch.randn(int(fourier-1),in_features_ori)), requires_grad=False)
        # self.register_parameter(name='w0', param=torch.nn.Parameter(torch.randn(int(fourier-1),in_features_ori)))
        # self.register_parameter(name='b0', param=torch.nn.Parameter(torch.randn(int(fourier-1),in_features_ori)))
        self.A0 = nn.Parameter(torch.nn.Parameter(torch.randn(int(fourier-1),in_features_ori)), requires_grad=trainble  )
        self.w0 = nn.Parameter(torch.nn.Parameter(torch.randn(int(fourier-1),in_features_ori)), requires_grad=trainble  )
        self.b0 = nn.Parameter(torch.nn.Parameter(torch.randn(int(fourier-1),in_features_ori)), requires_grad=trainble  )
        print('[X] in_features',in_features)
        print('[X] in_features_ori',in_features_ori)
        print('[X] fourier',fourier)
        print('[X] trainble',trainble)
        # print('[X] (fourier-1)*in_features_ori',(fourier-1)*in_features_ori)
        print('')

    def forward_mu(self, inputs, context=None):
        # print(inputs.max((0)))
        # print(self.w0)
        # x_proj = (2. * np.pi * x) @ B.t()

        # print(self.A0)
        # print(inputs.shape)
        # print(self.A0.shape)
        # if self.matrix:
            # (self.A0 @ inputs.t())
        
        if self.matrix is True:
            nA0 = (self.A0 @ inputs.t()).t() 
            nw0 = (self.w0 @ inputs.t()).t() 
            nb0 = (self.b0 @ torch.ones_like(inputs).t()).t()
            ninput =  nA0* torch.sin(nw0+nb0)
            inputs = torch.cat((inputs,ninput), dim=1)

        # new_tensor = torch.cat(inputs.shape[0]*[self.A0])
        # new_tensor = self.A0.repeat(inputs.shape[0], 1, 1).t()*inputs
        # print('new_tensor',new_tensor.shape)
        # new_tensor torch.Size([12700, 27])
        # ninput = self.A0.repeat(inputs.shape[0], 1, 1)*torch.sin(self.w0.repeat(inputs.shape[0], 1, 1)*inputs+self.b0.repeat(inputs.shape[0], 1, 1))
        # print('ninput',ninput.shape)

        # print(  (self.A0[0,:]*inputs).shape )
        # print(self.A0.shape[0])
        # print( ' torch.cat((inputs, self.A0[0,:]*inputs), dim=1)', torch.cat((inputs, self.A0[0,:]*inputs), dim=1))
        else:
            for jj in range(self.A0.shape[0]):
                inputs = torch.cat((inputs, self.A0[jj,:]*torch.sin(self.w0[jj,:]*inputs[:,:self.A0.shape[1]]+self.b0[jj,:])  ), dim=1)

        # print(torch.sin(inputs).shape)
        # print(torch.sin(self.w0*inputs).shape)
        # print('after',inputs.shape)


        # inputs = torch.cat((inputs, 3*self.A0*torch.sin(self.w0*inputs+self.b0)), dim=1)


        # print(inputs.max())
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        
        # Pass through output layers
        mu = self.output_mu(temps)
        return mu

    def obtain_samples(self, x, nsamples):
        # Forward pass
        x = (x-self.x_mean)/self.x_std
        x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        dist = self.forward_mu(x)
        return torch.unsqueeze(dist, 0).detach().numpy()*self.y_std + self.y_mean

    def evaluate(self, x):
        # y = (y-self.y_mean)/self.y_std
        # print('---------')
        # print(self.w0)
        # print(self.A0)
        # print(self.b0)
        # print('---------')

        if torch.is_tensor(x) is False: x = torch.from_numpy(x)#.to(device)
        if len(x.shape)<1.5:
            x = x.unsqueeze(0)
        x = (x-self.x_mean)/self.x_std
        # x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        if x.shape[0] ==1:
            return self.forward_mu(x)[0,:].detach().numpy()*self.y_std + self.y_mean
        else:
            return self.forward_mu(x).detach().numpy()*self.y_std + self.y_mean

    def forward_mse(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        mu = self.forward_mu(x)
        return self.loss_function_mse(y, mu)

    def loss_function_mse(self, y, dist):
        # return +torch.mean(torch.abs(y - dist))
        return self.l1loss(y,dist)








class ResidualNet5(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs. Only mu!"""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.elu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
        train_loader=None,
        input_encoder=None,
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
                    zero_initialization=zero_initialization,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_mu = nn.Linear(hidden_features, out_features)
        
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

        # self.l1loss = nn.L1Loss()
        self.l1loss = nn.MSELoss()

        if input_encoder is not None:
            print('[INFO] Using a pretrained input encoder')
            self.input_encoder = torch.load(input_encoder)
            for parami in self.input_encoder.parameters():
                parami.requires_grad = False
            self.y_std = self.input_encoder.y_std
            self.y_mean = self.input_encoder.y_mean
            self.x_std = self.input_encoder.x_std
            self.x_mean = self.input_encoder.x_mean
            self.input_encoder.x_std = 1.0
            self.input_encoder.x_mean = 0.0
            self.input_encoder.y_std = 1.0
            self.input_encoder.y_mean = 0.0

    def forward_mu(self, inputs, context=None):
        sinputs = torch.from_numpy(self.input_encoder.evaluate(inputs))

        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        
        # Pass through output layers
        mu = self.output_mu(temps)
        return mu+sinputs

    def obtain_samples(self, x, nsamples):
        # Forward pass
        x = (x-self.x_mean)/self.x_std
        x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        dist = self.forward_mu(x)
        return torch.unsqueeze(dist, 0).detach().numpy()*self.y_std + self.y_mean

    def evaluate(self, x):
        # y = (y-self.y_mean)/self.y_std
        if torch.is_tensor(x) is False: x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        x = (x-self.x_mean)/self.x_std
        # x = torch.from_numpy(x).unsqueeze(0)#.to(device)
        return self.forward_mu(x).detach().numpy()*self.y_std + self.y_mean

    def forward_mse(self, y, x):
        y = (y-self.y_mean)/self.y_std
        x = (x-self.x_mean)/self.x_std
        mu = self.forward_mu(x)
        return self.loss_function_mse(y, mu)

    def loss_function_mse(self, y, dist):
        # return +torch.mean(torch.abs(y - dist))
        return self.l1loss(y,dist)

import numpy as np
import torch
import torch.nn.functional  as F
from torch import nn
from torch.autograd import Variable
from torch.nn import init
import sys


def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


class LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type='mse'):
        if rec_type == 'mse':
            if dropout_mask is None:
                loss = torch.mean((real - predicted).pow(2))
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(dropout_mask)
        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none').mean()
        else:
            raise Exception
        return loss
    
    def weighted_reconstruction_loss(self, real, predicted, weight, dropout_mask=None, rec_type='mse'):
        if rec_type == 'mse':
            if dropout_mask is None:
                weighted_loss = torch.sum((real - predicted).pow(2), 1) * weight
                loss = torch.sum(weighted_loss) / (real.size()[0]*real.size()[1])
                # print(f"mse loss weighted: ", loss)
                # print(f"mse loss old: ", torch.mean((real - predicted).pow(2)))
            else:
                # TODO add weights
                # loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(dropout_mask)
                weighted_loss = torch.sum((real - predicted).pow(2) * dropout_mask, 1) * weight
                loss = torch.sum(weighted_loss) / torch.sum(dropout_mask)

        elif rec_type == 'bce':
            loss = F.binary_cross_entropy(predicted, real, reduction='none').mean()
        else:
            raise Exception
        return loss
    
    def new_KL_loss(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
    
    def weighted_new_KL_loss(self, mu, logvar, weight):
        weighted_KL = torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) * weight
        KLD = -0.5 * torch.sum(weighted_KL)

        return KLD

    def log_normal(self, x, mu, var):

        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).cuda()).sum(0) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))


class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, ):
        y = self.gumbel_softmax_sample(logits, temperature)
        return y

    def forward(self, x, temperature=1.0):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature)
        return logits, prob, y


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu.squeeze(2), logvar.squeeze(2)


class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, y_pos_dim, y_prime_dim, n_gene, nonLinear):
        super(InferenceNet, self).__init__()
 
        self.inference_x_encoder = torch.nn.ModuleList([
            nn.Linear(x_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            # nonLinear,
            # nn.Linear(z_dim, z_dim)
        ])
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(z_dim + y_pos_dim + y_prime_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            Gaussian(z_dim, 1)   
        ])

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std

        return z

    def qzxy(self, x, y, y_prime, adj):
        '''
        x: (bs, n_gene, 1)
        y: (bs, y_feature_dim)
        '''
        # x encoder
        # (bs, n_gene, 1) -> (bs, n_gene, z_dim)
        for layer in self.inference_x_encoder:
            x = layer(x)
        # print(f"x size after encoded: {x.size()}")

        # x: (bs, n_gene, z_dim) -> (z_dim, bs, n_gene)
        x = torch.matmul(x.permute(2, 0, 1), adj)  # broadcast on z_dim results (z, bs, m)
        x = x.permute(1, 2, 0)  # Permute back to get (bs, m, z)
        # print(f"x size after mul with adj: {x.size()}")

        # y repeat: (bs, y_dim) -> (bs, n_gene, y_dim)
        y = y.unsqueeze(1).repeat(1, x.shape[1], 1)
        y_prime = y_prime.unsqueeze(1).repeat(1, x.shape[1], 1)
        # print(f"y repeat size: {y.size()}")
        concat = torch.cat((x, y, y_prime), dim=2)
        # print(f"x concat y size: {concat.size()}")

        for layer in self.inference_qzyx:
            concat = layer(concat)
        # print(f"mu size {concat[0].size()}, logvar size {concat[1].size()}")
        return concat

    def forward(self, x, y_pos_feature, y_prime_feature, adj, temperature=1.0):
        mu, logvar = self.qzxy(x, y_pos_feature, y_prime_feature, adj)
        var = torch.exp(logvar)

        z = self.reparameterize(mu, var)
        # print(f"z size: {z.size()}")

        output = {'mean': mu, 'logvar': logvar, 'var': var, 'gaussian': z}
        return output


class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, y_pos_dim, y_prime_dim, n_gene, nonLinear):
        super(GenerativeNet, self).__init__()
        self.n_gene = n_gene

        self.generative_pxzy = torch.nn.ModuleList([
            nn.Linear(1 + y_pos_dim + y_prime_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
            nonLinear,
            nn.Linear(z_dim, z_dim),
        ])

        self.generative_x_decoder = torch.nn.ModuleList([
            nn.Linear(z_dim, z_dim),
            nonLinear,
            # nn.Linear(z_dim, z_dim),
            # nonLinear,
            nn.Linear(z_dim, x_dim),
        ])

    def pxzy(self, z, y, y_prime):
        # z: (bs, n_gene)
        # y: (bs, y_dim)
        z = z.unsqueeze(2)  # (bs, n_gene) -> (bs, n_gene, 1)
        y = y.unsqueeze(1).repeat(1, z.shape[1], 1) # (bs, y_dim) -> (bs, n_gene, y_dim)
        y_prime = y_prime.unsqueeze(1).repeat(1, z.shape[1], 1)
        concat = torch.cat((z, y, y_prime), dim=2)
        # print(f"z+y concat size: {concat.size()}")

        for layer in self.generative_pxzy:
            concat = layer(concat)
        return concat
    
    def x_decoder(self, x):
        for layer in self.generative_x_decoder:
            x = layer(x)
        return x

    def forward(self, z, y, y_prime, adj_inv):
        
        x = self.pxzy(z, y, y_prime)  
        # print(f"x after pxzy: {x.size()}")
        
        # x: (bs, n_gene, z_dim) -> (z_dim, bs, n_gene)
        x = torch.matmul(x.permute(2, 0, 1), adj_inv)  # broadcast on z_dim results (z, bs, m)
        x = x.permute(1, 2, 0)  # Permute back to get (bs, m, z)
        # x: (bs, n_gene, z_dim)
        # print(f"x after mul adj_inv: {x.size()}")

        x_rec = self.x_decoder(x)   
        x_rec = torch.squeeze(x_rec, 2)  # (bs, n_gene, 1) -> (bs, n_gene)
        # print(f"x_rec (after squeeze): {x_rec.size()}")
        output = {'x_rec': x_rec}

        return output


class Encoder_Y_Net(nn.Module):
    def __init__(self, y_pos_dim, nonLinear):
        super(Encoder_Y_Net, self).__init__()

        self.y_encoder = torch.nn.ModuleList([
            nn.Linear(2, 64),
            nonLinear,
            nn.Linear(64, y_pos_dim)
        ])
    
    def forward(self, y_pos):
        # directly apply a NN on position (x,y)
        for layer in self.y_encoder:
            y_pos = layer(y_pos)

        return y_pos


class Encoder_Yprime_Net(nn.Module):
    def __init__(self, y_prime_input_dim, y_prime_dim, nonLinear):
        super(Encoder_Yprime_Net, self).__init__()
        hidden_dim = 128 if y_prime_input_dim > 128 else 64

        self.y_prime_encoder = torch.nn.ModuleList([
            nn.Linear(y_prime_input_dim, hidden_dim),
            nonLinear,
            nn.Linear(hidden_dim, y_prime_dim)
        ])

    def forward(self, y_prime):
        for layer in self.y_prime_encoder:
            y_prime = layer(y_prime)

        return y_prime


class CVAE_EAD_newED(nn.Module):
    def __init__(self, adj_A, x_dim, z_dim, y_dim, y_pos_dim, y_prime_dim, y_prime_input_dim=2):
        super(CVAE_EAD_newED, self).__init__()
        # adj_A_init, x_dim = 1, z_dim=opt.n_hidden=128, opt.K=1
        self.y_pos_dim = y_pos_dim    # dim of the y encoded feature
        self.y_prime_input_dim = y_prime_input_dim
        self.y_prime_dim = y_prime_dim
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True, name='adj_A'))
        self.n_gene = n_gene = len(adj_A)
        nonLinear = nn.Tanh()
        self.encoder_Y = Encoder_Y_Net(self.y_pos_dim, nonLinear)
        self.encoder_Yprime = Encoder_Yprime_Net(self.y_prime_input_dim, self.y_prime_dim, nonLinear)
        self.inference = InferenceNet(x_dim, z_dim, y_dim, y_pos_dim, self.y_prime_dim, n_gene, nonLinear)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim, y_pos_dim, self.y_prime_dim, n_gene, nonLinear)
        self.losses = LossFunctions()
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def _one_minus_A_t(self, adj):
        adj_normalized = torch.from_numpy(np.eye(adj.shape[0])).float() - adj.transpose(0, 1)
        return adj_normalized
    
    def _one_minus_A_t_gpu(self, adj):
        adj_normalized = torch.from_numpy(np.eye(adj.shape[0])).float().cuda() - adj.transpose(0, 1)
        return adj_normalized
    

    def forward(self, x, Y_pos, Y_prime, loss_weight=None, dropout_mask=None, temperature=1.0, opt=None):
        # x: input, Y_pos: condition (new)
        x_ori = x
        x = x.view(x.size(0), -1, 1)
        # print(f"x view shape: {x.size()}")

        # use mask to not update diagonal of adj_A
        if opt.GPU:
            mask = Variable(torch.from_numpy(np.ones(self.n_gene) - np.eye(self.n_gene)).float(), requires_grad=False).cuda()
            adj_A_t = self._one_minus_A_t_gpu(self.adj_A * mask) # (I-A^T)
        else:
            mask = Variable(torch.from_numpy(np.ones(self.n_gene) - np.eye(self.n_gene)).float(), requires_grad=False)
            adj_A_t = self._one_minus_A_t(self.adj_A * mask)  # (I-A^T)
          
        adj_A_t_inv = torch.inverse(adj_A_t)        # (I-A^T)^{-1}

        Y_pos_feature = self.encoder_Y(Y_pos)
        Y_prime_feature = self.encoder_Yprime(Y_prime)
        # print(f"Y_pos_feature size: {Y_pos_feature.size()}")
        out_inf = self.inference(x, Y_pos_feature, Y_prime_feature, adj_A_t, temperature)

        z = out_inf['gaussian']

        out_gen = self.generative(z, Y_pos_feature, Y_prime_feature, adj_A_t_inv)

        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        dec = output['x_rec']

        if loss_weight is None:
            loss_rec = self.losses.reconstruction_loss(x_ori, output['x_rec'], dropout_mask, 'mse')
            # print(f"loss_rec: {loss_rec}")
            loss_KL = self.losses.new_KL_loss(output['mean'], output['logvar']) * opt.beta
            # print(f"loss_KL: {loss_KL}")
        else:
            # need loss weight
            loss_rec = self.losses.weighted_reconstruction_loss(x_ori, output['x_rec'], loss_weight, dropout_mask, 'mse')
            # print(f"weighted loss_rec: {loss_rec}")
            loss_KL = self.losses.weighted_new_KL_loss(output['mean'], output['logvar'], loss_weight) * opt.beta
            # print(f"weighted loss_KL: {loss_KL}")
        
        loss = loss_rec + loss_KL

        return loss, loss_rec, loss_KL, dec, output['mean']




import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# function for f
from .diagonal_func import DiagonalFunc

class SDEModel(nn.Module):

    def __init__(self, N, device, context_size, gamma_size, f_nn_size, mean_T, std_T):
        super(SDEModel, self).__init__()

        self.N = N
        self.mean_T = mean_T
        self.std_T = std_T
        self.device = device
        # self.w_mask = (torch.ones(N,N) - torch.eye(N)).to(device) # mask for off-diagonal weights in network
        self.w_mask = torch.ones(N,N,N)
        for i in range(N):
            self.w_mask[i,:,:] *= (~torch.eye(N,dtype=bool)).type(torch.DoubleTensor)
            self.w_mask[:,i,:] *= (~torch.eye(N,dtype=bool)).type(torch.DoubleTensor)
            self.w_mask[:,:,i] *= (~torch.eye(N,dtype=bool)).type(torch.DoubleTensor)
        self.w_mask = self.w_mask.to(device)

        # diagonal neural net in dynamics
        self.f = DiagonalFunc(N, 2 + context_size, f_nn_size)

        # neural net for sigma_x
        self.sigma_nn = nn.Sequential(
            nn.Linear(N, N),
            nn.ELU(),
            nn.Linear(N, N),
            nn.Sigmoid()
        )
        
        # rnn and nn for mortality
        self.hazard1 = nn.GRU(N + 1, gamma_size, batch_first=True)
        self.hazard2 = nn.GRU(gamma_size, gamma_size - 15, batch_first=True)
        self.hazard_out = nn.Sequential(
            nn.ELU(),
            nn.Linear(gamma_size-15, 1)
        )

        # posterior drift nn
        self.g = nn.Sequential(
            nn.Linear(N + 1 + context_size, 8),
            nn.ELU(),
            nn.Linear(8, N, bias=False)
        )

    # output of sigma_x nn (this is the network for stochastic noise)
    def sigma_x(self, x): # lower bound of 1e-5
        return self.sigma_nn(x) + 1e-5
    
    # Args: 
    #   x -> Feature vector (size 29)
    #   z -> is an input to the diagonal_func
    #   W -> 3D interaction matrix
    # Purpose:
    #   prior drift of model
    def prior_drift(self, x, z, W):
        # return torch.matmul(x, self.w_mask*W) + self.f(x,z)
        M = x.shape[0]
        T = x.shape[1]
        # Wx = torch.zeros(M,T,self.N)
        # for i in range(self.N):
        #     for j in range(self.N):
        #         for k in range(self.N):
        #             for t in range(T):  
        #                 if len({i,j,k}) == 3: #all health variables are unique
        #                     Wx[:,t,i] += W[:,i,j,k]*(x[:,t,j]+ x[:,t,k])
        # return Wx/4 + self.f(x,z)

        # for i in range(self.N):
        #     # Wx[:,:,i] = torch.sum(torch.matmul(W[:,i,:,:],x),axis=-1)
        #     Wx[:,:,i] = torch.sum(torch.matmul(x,(self.w_mask*W)[:,i,:,:]),axis=-1)

        x_cols = (torch.ones(self.N,x.shape[0],x.shape[1],self.N, device=x.device)*x).permute(1,2,0,3) # every column has same values
        x_rows = x_cols.permute(0,1,3,2) # every row has same values
        x_star = x_cols + x_rows
        
        # for i in range(self.N):
        #     Wx[:,:,i] = torch.sum(x_star*(self.w_mask*W.unsqueeze(1))[:,:,i,:,:],dim=(-1,-2))
        Wx = torch.sum(x_star.unsqueeze(2) * (self.w_mask*W).unsqueeze(1),dim=(-1,-2))/self.N

        return Wx + self.f(x,z)

    # Args: 
    #   x -> Feature vector (size 29)
    #   z -> is an input to the diagonal_func
    #   W -> 3D interaction matrix
    # Purpose:
    #   posterior drift of model
    def posterior_drift(self, x, z, W):
        return torch.matmul(x, self.w_mask*W) + self.g(torch.cat((x,z),dim=-1)) + self.f(x,z)
    

    # Args: 
    #   x -> Feature vector (size 29)
    #   h -> previous hidden state for RNN
    # Purpose:
    #   output of survival rnn
    def log_Gamma(self, x, h):
        g, h1 = self.hazard1(x.unsqueeze(1), h[0])
        g, h2 = self.hazard2(g, h[1])
        h = (h1, h2)
        return self.hazard_out(g).squeeze(1), h


    # Args: 
    #         x -> Feature vector (size 29)
    #         h -> previous hidden state for RNN
    #         t -> ages for previous time set
    #   context -> background
    #         W -> 3D interaction matrix
    # Purpose:
    #   output one step of posterior SDE and survival model
    def forward(self, x, h, t, context, W):
        M = x.shape[0]

        z_RNN = torch.cat(((t.unsqueeze(-1) - self.mean_T)/self.std_T, context), dim=-1)
        x_ = x.clone()
        log_Gamma, h = self.log_Gamma(torch.cat((x, (t.unsqueeze(-1) - self.mean_T)/self.std_T),dim=-1), h)
        # dx = torch.matmul(x.unsqueeze(1), self.w_mask*W).squeeze(1) + self.f(x, z_RNN) + self.g(torch.cat((x,z_RNN),dim=-1))
        # for i in range(self.N):
        #     for j in range(self.N):
        #         for k in range(self.N):
        #             if len({i,j,k}) == 3: # all health variables are unique
        #                 Wx[:,i] += W[i,j,k]*(x[:,j]+x[:,k])
        # dx = Wx/4 + self.f(x,z_RNN) + self.g(torch.cat((x,z_RNN),dim=-1))
        
        # for i in range(self.N):
        #     Wx[:,i] = torch.sum(torch.matmul(x.unsqueeze(1),(self.w_mask*W)[i,:,:]).squeeze(1),axis=-1)
        
        # creating matrix with pairwise addition of each health variable
        x_cols = (torch.ones(self.N,x.shape[0],self.N, device=x.device)*x).permute(1,0,2) # every column has same values
        x_rows = x_cols.permute(0,2,1) # every row has same values
        x_star = x_cols + x_rows

        # for i in range(self.N):
        #     Wx[:,i] = torch.sum(x_star*(self.w_mask*W)[i,:,:],dim=(1,2))
        Wx = torch.sum(x_star.unsqueeze(1)*(self.w_mask*W),axis=(-1,-2))/self.N
        dx = Wx + self.f(x,z_RNN) + self.g(torch.cat((x,z_RNN),dim=-1))
        log_dS = -torch.exp(log_Gamma).reshape(x.shape[0])
        
        return dx, log_dS, log_Gamma, h, self.sigma_x(x_)
        
    # Args: 
    #         x -> Feature vector (size 29)
    #         h -> previous hidden state for RNN
    #         t -> ages for previous time set
    #   context -> background
    #         W -> 3D interaction matrix
    # Purpose:
    #   output one step of prior SDE and survival model
    def prior_sim(self, x, h, t, context, W):
        
        z_RNN = torch.cat(((t.unsqueeze(-1) - self.mean_T)/self.std_T, context), dim=-1)
        x_ = x.clone()

        log_Gamma, h = self.log_Gamma(torch.cat((x, (t.unsqueeze(-1) - self.mean_T)/self.std_T),dim=-1), h)
        dx = torch.matmul(x.unsqueeze(1), self.w_mask*W).squeeze(1) + self.f(x_, z_RNN)
        log_dS = -torch.exp(log_Gamma).reshape(x.shape[0])
        
        return dx, log_dS, log_Gamma, h, self.sigma_x(x_)

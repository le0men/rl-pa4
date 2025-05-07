import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousPolicy(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        self.x_dim = obs_dim
        self.a_dim = act_dim

        self.linear_1 = nn.Linear(self.x_dim, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, self.a_dim)

        self.log_stds = nn.Parameter(torch.full((self.a_dim,), -0.5))

    def _mean_std(self, x):
        h = F.leaky_relu(self.linear_1(x))
        h = F.leaky_relu(self.linear_2(h))
        means = self.linear_3(h)
        stds = torch.exp(self.log_stds).expand_as(means)
        return means, stds

    def _gaussian_logp(self, means, stds, acts):
        cov = torch.diag_embed(stds**2) 
        dist = torch.distributions.MultivariateNormal(means, cov)
        return dist.log_prob(acts)


    def forward(self, x, *, with_logp=False, smooth=False):
        means, stds = self._mean_std(x)
        acts = torch.normal(means, stds)

        if smooth:  # optional exploration noise
            acts += torch.normal(torch.zeros_like(acts), torch.ones_like(acts) * 2.0)

        if with_logp:
            logp = self._gaussian_logp(means, stds, acts)
            return acts, logp
        return acts

    def sample_action(self, x, *, smooth=False):
        return self.forward(x,smooth=smooth)
        

    def compute_log_likelihood(self, x, acts):
        means, stds = self._mean_std(x)    
        return self._gaussian_logp(means,stds,acts)


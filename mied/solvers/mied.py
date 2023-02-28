import torch
import numpy as np
import math
from mied.solvers.particle_base import ParticleBasedSolver

def log_exp_diff(a, b):
    '''
    Compute log|e^a - e^b| * sign(a-b)
    :param a, b: torch scalars
    '''

    if a > b:
        return a + torch.log(1 - torch.exp(b - a))
    else:
        return -(b + torch.log(1 - torch.exp(a - b)))



class MIED(ParticleBasedSolver):
    def __init__(self, *,
                 kernel,
                 eps,
                 riesz_s,
                 alpha_mied,
                 include_diag,
                 diag_mul=1.3,
                 **kwargs):
        '''
        Descent on energy
            E(\mu) = \iint \phi_\eps(x-y) (p(x)p(y))^{-\alpha} \dd\mu(x)\dd\mu(y)
        :param kernel: ['riesz', 'gaussian', 'laplace']
        :param eps: epsilon in the kernel
        :param include_diag: ['ignore', 'include', 'diag_only', 'nnd']
        '''
        super().__init__(**kwargs)

        self.device = self.problem.device
        if kernel in ['gaussian', 'laplace']:
            assert(eps >= 1e-6)
        embed_dim = self.problem.get_embed_dim()
        self.embed_dim = embed_dim
        self.alpha = alpha_mied
        if kernel == 'riesz' and riesz_s < 0:
            assert(self.alpha >= 0.5) # requirement for hypersingular riesz energy
            riesz_s = 2 * self.alpha * embed_dim + 1e-4
        self.kernel = kernel
        self.eps = eps
        self.riesz_s = riesz_s
        self.include_diag = include_diag
        self.diag_mul = diag_mul


    def compute_energy(self, X):
        '''
        :param X: (B, D)
        :return: a scalar, the weighted riesz energy
        '''

        log_p = self.problem.eval_log_p(X) # (B,)

        B = X.shape[0]
        diff = X.unsqueeze(1) - X.unsqueeze(0) # (B, B, D)
        diff_norm_sqr = diff.square().sum(-1) # (B, B)

        if self.include_diag == 'nnd_scale':
            vals, _ = torch.topk(diff_norm_sqr, 2, dim=-1, largest=False)
            vals = vals.detach()[:, 1]

            # Use \phi(h_i / (1.3d)^{1/d}) for the diagonal term.
            vals = vals / math.pow(self.diag_mul * self.embed_dim, 2.0 / self.embed_dim)
            diff_norm_sqr = diff_norm_sqr + torch.diag(vals)

        if self.kernel == 'gaussian':
            # \phi(x-y) = \exp(-||x-y||^2/(2 * eps))
            tmp = -diff_norm_sqr / (2 * self.eps)
        elif self.kernel == 'laplace':
            tmp = -(diff_norm_sqr + 1e-10).sqrt() / self.eps
        else:
            assert(self.kernel == 'riesz')
            log_dist_sqr = (diff_norm_sqr + self.eps).log() # (B, B)
            tmp = log_dist_sqr * -self.riesz_s / 2

        tmp2 = (log_p.unsqueeze(1) + log_p.unsqueeze(0)) # (B, B)
        tmp2 = tmp2 * -self.alpha # (B, B)

        tmp = tmp + tmp2

        mask = torch.eye(B, device=X.device, dtype=torch.bool) # (B, B)
        mask = torch.logical_not(mask) # (B, B)

        if self.include_diag != 'ignore':
            mask = torch.logical_or(mask,
                                    torch.eye(B, device=X.device,
                                              dtype=torch.bool))
        else:
            mask = torch.eye(B, device=X.device, dtype=torch.bool) # (B, B)

        mask = mask.reshape(-1)
        tmp = tmp.reshape(-1)
        tmp = torch.masked_select(tmp, mask)

        energy = torch.logsumexp(tmp, 0) # scalar
        # if self.include_diag in ['include', 'nnd']:
        #     energy = energy + -2 * math.log(B)
        # else:
        #     energy = energy + -2 * math.log(B - 1)

        return energy


    def step(self, i):
        if self.optimizer_conf['cls'] == 'LBFGS':
            def closure():
                self.optimizer.zero_grad()
                X = self.problem.reparametrize(self.particles)

                # Subclass must have a compute_energy function.
                F = self.compute_energy(X)
                self.last_F = F.item()
                F.backward()
                return F
            self.optimizer.step(closure)
        else:
            super().step(i)


    def compute_update(self, i, X):
        '''
        :return: (B, D)
        '''
        F = self.compute_energy(X) # scalar
        self.last_F = F.item()
        grad_F = torch.autograd.grad(F, X)[0] # (B, D)

        return -grad_F


    def custom_post_step(self, i):
        return {
            'Riesz energy': self.last_F
        }


    def get_progress_msg(self):
        return 'E: {:6f}, G_vio: {:6f}'.format(self.last_F,
                                               self.projector.get_violation())


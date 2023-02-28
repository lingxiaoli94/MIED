import torch
import numpy as np

from mied.utils.batch_jacobian import compute_jacobian
from mied.utils.kernels import GaussianKernel
from mied.solvers.particle_base import ParticleBasedSolver

def compute_ksd(X, grad_log_p, kernel):
    '''
    :param X: (B, D)
    :param grad_log_p: (B, D)
    :param kernel: an instance of KernelBase, assumed to be symmetric
    :return: a scalar, the kernel stein discrepancy
    '''
    B, D = X.shape

    X_ex1 = X.unsqueeze(1).expand(-1, B, -1).reshape(-1, D) # (BB, D)
    X_ex2 = X.unsqueeze(0).expand(B, -1, -1).reshape(-1, D) # (BB, D)
    score_ex1 = grad_log_p.unsqueeze(1).expand(-1, B, -1).reshape(-1, D)
    score_ex2 = grad_log_p.unsqueeze(0).expand(B, -1, -1).reshape(-1, D)

    k = kernel.eval(X_ex1, X_ex2) # (BB,)
    grad_1_k = kernel.grad_1(X_ex1, X_ex2) # (BB, D)
    grad_2_k = kernel.grad_1(X_ex2, X_ex1) # (BB, D)
    div_2_grad_1_k = kernel.div_2_grad_1(X_ex1, X_ex2) # (BB,)

    tmp = (score_ex1 * score_ex2).sum(-1) * k # (BB,)
    tmp = tmp + (score_ex1 * grad_2_k).sum(-1)
    tmp = tmp + (score_ex2 * grad_1_k).sum(-1)
    tmp = tmp + div_2_grad_1_k

    return tmp.mean()


class KSDD(ParticleBasedSolver):
    def __init__(self,
                 sigma=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel = GaussianKernel(sigma=sigma)


    def compute_update(self, i, X):
        '''
        :return: (B, D)
        '''
        log_p = self.problem.eval_log_p(X) # (B,)
        # Note: KSDD requires second-order derivatives of log_p.
        grad_log_p = compute_jacobian(log_p.unsqueeze(-1), X,
                                      create_graph=True, retain_graph=True)
        grad_log_p = grad_log_p.squeeze(-2) # (B, D)

        F = compute_ksd(X, grad_log_p, self.kernel)
        self.last_F = F.item()
        grad_F = torch.autograd.grad(F, X)[0] # (B, D)
        return -grad_F


    def custom_post_step(self, i):
        return {
            'KSD': self.last_F
        }


    def get_progress_msg(self):
        return 'KSD: {:6f}, G_vio: {:6f}'.format(
            self.last_F, self.projector.get_violation())

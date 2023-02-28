import torch
import numpy as np

from mied.solvers.particle_base import ParticleBasedSolver
from mied.utils.batch_jacobian import compute_jacobian


def svgd_update(P, grad_log_p, kernel='gaussian', kernel_h=-1,
                riesz_s=-1, riesz_eps=1e-4):
    '''
    SVGD update with Gaussian kernel.

    :param P: (B, D)
    :return: update direction, (B, D)
    '''
    assert(not P.isnan().any())
    assert(not grad_log_p.isnan().any())

    n = P.shape[0]

    P_diff = P.unsqueeze(1) - P.unsqueeze(0) # (B, B, D)
    dist_sqr = P_diff.square().sum(-1) # (B, B)

    if kernel == 'gaussian':
        if kernel_h < 0:
            mean_dist_sqr = dist_sqr.reshape(-1).median()
            h = mean_dist_sqr / (np.log(n) + 1e-6)
        else:
            h = kernel_h

        K = torch.exp(- dist_sqr / (h + 1e-8)) # (B, B)
        grad_K = 2 * K.unsqueeze(-1) * P_diff / (h + 1e-8) # (B, B, D)
    else:
        assert(kernel == 'riesz')
        if riesz_s < 0:
            riesz_s = P.shape[-1] + 1.0
        K = torch.pow(dist_sqr + riesz_eps, -riesz_s / 2)
        grad_K = (riesz_s/2) * torch.pow(
            dist_sqr + riesz_eps, -riesz_s / 2 - 1).unsqueeze(-1) * 2 * P_diff

    '''
    phi(x_i) = 1/n * \sum_j k(x_i, x_j) grad_log_p(x_j) + grad_K_x_j(x_i, x_j)
    '''
    Phi = K.unsqueeze(-1) * grad_log_p.unsqueeze(0) + grad_K # (B, B, D)
    Phi = Phi.mean(1) # (B, D)

    assert(not Phi.isnan().any())

    return Phi


class SVGD(ParticleBasedSolver):
    def __init__(self,
                 kernel_h=-1,
                 kernel='gaussian',
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.kernel_h = kernel_h


    def compute_update(self, i, X):
        log_p = self.problem.eval_log_p(X) # (B,)

        grad_log_p = compute_jacobian(log_p.unsqueeze(-1), X,
                                      create_graph=False, retain_graph=False)
        grad_log_p = grad_log_p.squeeze(-2)
        update = svgd_update(X, grad_log_p, kernel=self.kernel,
                             kernel_h=self.kernel_h)
        self.last_update_norm = update.square().sum().item()

        P_diff = X.unsqueeze(1) - X.unsqueeze(0) # (B, B, D)
        dist_sqr = P_diff.square().sum(-1) # (B, B)
        self.last_med_dist = dist_sqr.reshape(-1).median().sqrt().item()
        return update


    def custom_post_step(self, i):
        return {
            'Update norm': self.last_update_norm,
            'Median': self.last_med_dist,
        }


    def get_progress_msg(self):
        return 'Norm: {:6f}, G_vio: {:6f}'.format(
            self.last_update_norm, self.projector.get_violation())

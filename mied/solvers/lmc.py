import torch
import numpy as np

from mied.solvers.particle_base import ParticleBasedSolver
from mied.utils.batch_jacobian import compute_jacobian
from mied.solvers.mirror_maps import BoxMap, BoxEntropicMap

class LMC(ParticleBasedSolver):
    def __init__(self,
                 lmc_lr,
                 mirror_map,
                 **kwargs):
        super().__init__(direct_update=True,
                         **kwargs)
        assert(self.optimizer_conf['cls'] == 'SGD' and
               self.optimizer_conf['lr'] == 1.0)
        self.lr = lmc_lr

        if mirror_map == 'box':
            self.mirror_map = BoxMap()
        elif mirror_map == 'box_entropic':
            self.mirror_map = BoxEntropicMap()
        else:
            raise Exception(f'Unknown mirror map: {mirror_map}')


    def compute_update(self, i, X):
        B, D = X.shape
        log_p = self.problem.eval_log_p(X) # (B,)

        grad_log_p = compute_jacobian(log_p.unsqueeze(-1), X,
                                      create_graph=False, retain_graph=False)
        grad_log_p = grad_log_p.squeeze(-2) # (B, D)

        Z = self.mirror_map.nabla_phi(X) # (B, D)
        xi = torch.randn([B, D], device=X.device) # (B, D)

        drift = np.sqrt(2 * self.lr) * self.mirror_map.nabla2_phi_sqrt_mul(X, xi) # (B, D)

        Z_new = Z + self.lr * grad_log_p + drift
        return self.mirror_map.nabla_phi_star(Z_new)

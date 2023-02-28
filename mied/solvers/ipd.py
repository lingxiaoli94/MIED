import torch
import numpy as np

from mied.solvers.particle_base import ParticleBasedSolver
from mied.utils.batch_jacobian import compute_jacobian


'''
Independent particle descent, a dumb baseline.
'''
class IPD(ParticleBasedSolver):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)


    def compute_update(self, i, X):
        log_p = self.problem.eval_log_p(X) # (B,)
        self.last_log_p = log_p.mean()

        grad_log_p = compute_jacobian(log_p.unsqueeze(-1), X,
                                      create_graph=False, retain_graph=False)
        update = grad_log_p.squeeze(1)
        return update


    def get_progress_msg(self):
        return 'log_p: {:6f}, G_vio: {:6f}'.format(
            self.last_log_p, self.projector.get_violation())

import torch
from mied.solvers.projector_base import ProjectorBase
from mied.utils.batch_jacobian import compute_jacobian
from mied.utils.proj_polyhedra import proj_polyhedra

'''
Handle multiple constraints by projecting the gradients using
Dystra algorithm.
'''
class DynamicBarrier(ProjectorBase):
    def __init__(self, *,
                 alpha_db=1.0,
                 merge_eq=True,
                 max_proj_itr=20):

        self.alpha = alpha_db
        self.merge_eq = merge_eq
        self.max_proj_itr = max_proj_itr
        self.violation = 0.0


    def step(self, X, update, problem):
        G = problem.eval_ineq(X) # (B, N_ineq)
        H = problem.eval_eq(X) # (B, N_eq)
        B = X.shape[0]

        if G is None and H is None:
            v = update
        else:
            if G is None:
                G = torch.zeros([B, 0]).to(X)
            if H is None:
                H = torch.zeros([B, 0]).to(X)

            if self.merge_eq:
                G = torch.cat([G, H.square().sum(-1, keepdim=True)], -1)
            else:
                G = torch.cat([G, H.square()], -1)
            self.violation = G.relu().sum()

            grad_G = compute_jacobian(G, X, create_graph=True,
                                      retain_graph=True) # (B, N_ineq, D)
            barrier = -self.alpha * G # (B, N_ineq)

            # Constraints are grad_G^T v <= barrier
            v = proj_polyhedra(update, grad_G, barrier,
                               max_num_itr=self.max_proj_itr) # (B, D)
        return v


    def get_violation(self):
        return self.violation

import torch
from mied.problems.problem_base import ProblemBase

class Analytical(ProblemBase):
    def __init__(self, *,
                 bbox,
                 embed_dim,
                 log_p_fn,
                 prior_sample_fn,
                 eq_fn=None,
                 ineq_fn=None,
                 reparam_fn=None,
                 gt_sample_fn=None,
                 **kwargs):
        '''
        A base class for simple analytical functions.

        :param bbox: a (D, 2) tensor, used only for evaluation.
        '''
        super().__init__(**kwargs)
        self.bbox = bbox
        self.embed_dim = embed_dim
        self.log_p_fn = log_p_fn
        self.eq_fn = eq_fn
        self.ineq_fn = ineq_fn
        self.prior_sample_fn = prior_sample_fn
        self.gt_sample_fn = gt_sample_fn
        self.reparam_fn = reparam_fn


    def get_embed_dim(self):
        return self.embed_dim


    def eval_log_p(self, P):
        return self.log_p_fn(P)


    def sample_prior(self, batch_size):
        return self.prior_sample_fn(batch_size, self.device)


    def reparametrize(self, Z):
        if self.reparam_fn is None:
            return super().reparametrize(Z)
        return self.reparam_fn(Z)


    def sample_gt(self, batch_size, refresh):
        if self.gt_sample_fn is not None:
            return self.gt_sample_fn(batch_size, refresh)
        return None


    def eval_eq(self, P):
        if self.eq_fn is None:
            return None
        return self.eq_fn(P)


    def eval_ineq(self, P):
        if self.ineq_fn is None:
            return None
        return self.ineq_fn(P)



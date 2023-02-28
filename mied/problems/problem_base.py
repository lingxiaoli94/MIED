from abc import ABC, abstractmethod
import torch
from mied.utils.batch_jacobian import compute_jacobian

class ProblemBase(ABC):
    def __init__(self, *,
                 device,
                 in_dim):
        '''
        A problem describes the sampling problem with unnormalized density
        p(x) and constraints g(x) <= 0, h(x) = 0.

        :param device: used to generate ambient samples
        :param in_dim: ambient dimension
        '''
        self.device = device
        self.in_dim = in_dim


    @abstractmethod
    def sample_prior(self, batch_size):
        '''
        Sample prior particles (before applying reparameterization).
        '''
        pass


    @abstractmethod
    def get_embed_dim(self):
        pass


    @abstractmethod
    def eval_log_p(self, P):
        '''
        Evaluate the log density.
        - For sampling, can ignore the constant.
        - For evaluation purpose however the constant should be included
        if possible.

        :param P: positions, tensor of size (batch_size, in_dim)
        :return: (batch_size,)
        '''
        pass


    def sample_gt(self, batch_size, refresh):
        '''
        Implement this in cases where the problem can be sampled (for evaluation).
        '''
        return None


    def reparametrize(self, Z):
        '''
        :param Z: (B, Z)
        :return: (B, D)
        '''
        return Z


    def eval_eq(self, P):
        '''
        Evaluate the constraints h(x).

        :param P: positions, tensor of size (batch_size, in_dim)
        :return: (batch_size, num_eq)
        '''
        return None


    def eval_ineq(self, P):
        '''
        Evaluate the constraints g(x).

        :param P: positions, tensor of size (batch_size, in_dim)
        :return: (batch_size, num_ineq)
        '''
        return None


    def custom_eval(self, samples):
        return {}


    def custom_post_step(self, samples):
        return {}


from abc import ABC, abstractmethod
import torch
from mied.utils.batch_jacobian import compute_jacobian

class KernelBase(ABC):
    @abstractmethod
    def eval(self, X, Y):
        '''
        :param X: (B, D)
        :param Y: (B, D)
        :return: (B,)
        '''
        pass


    @abstractmethod
    def grad_1(self, X, Y):
        '''
        :param X: (B, D)
        :param Y: (B, D)
        :return: (B, D)
        '''
        pass


    @abstractmethod
    def div_2_grad_1(self, X, Y):
        '''
        :param X: (B, D)
        :param Y: (B, D)
        :return: (B,)
        '''
        pass



class GaussianKernel(KernelBase):
    def __init__(self, sigma):
        '''
        k(x, y) = exp(-||x-y||^2/(2 sigma))
        :param sigma:
        '''
        self.sigma = sigma


    def eval(self, X, Y):
        return torch.exp(-(X - Y).square().sum(-1) / (self.sigma * 2))


    def grad_1(self, X, Y):
        return -(X - Y) / self.sigma * self.eval(X, Y).unsqueeze(-1)


    def div_2_grad_1(self, X, Y):
        D = X.shape[-1]
        return self.eval(X, Y) * (-(X - Y).square().sum(-1) / (self.sigma ** 2)
                                  + D / self.sigma)

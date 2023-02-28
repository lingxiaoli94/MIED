from abc import ABC, abstractmethod
import torch

def safe_log(x):
    # return torch.log(torch.maximum(1e-32, x))
    # return torch.log(torch.maximum(1e-8, x))
    return torch.log(x + 1e-32)


class MirrorMapBase(ABC):
    @abstractmethod
    def phi(self, theta):
        pass

    @abstractmethod
    def nabla_phi(self, theta):
        pass

    @abstractmethod
    def nabla_phi_star(self, eta):
        pass

    @abstractmethod
    def nabla2_phi_sqrt_mul(self, theta, rhs):
        pass


class BoxMap(MirrorMapBase):
    def phi(self, theta):
        return (-safe_log(1-theta) - safe_log(1+theta)).sum(-1)

    def nabla_phi(self, theta):
        return 1 / (1 - theta) - 1 / (1 + theta)

    def nabla_phi_star(self, eta):
        return ((1 + eta.square()).sqrt() - 1) / eta

    def nabla2_phi_sqrt_mul(self, theta, rhs):
        diag = 1 / (1-theta).square() + 1 / (1+theta).square()
        diag = diag.sqrt()

        return diag * rhs


class BoxEntropicMap(MirrorMapBase):
    def phi(self, theta):
        return (1 + theta) * safe_log(1 + theta) + (1 - theta) * safe_log(1 - theta)

    def nabla_phi(self, theta):
        return safe_log(1 + theta) - safe_log(1 - theta)

    def nabla_phi_star(self, eta):
        return 1 - 2 / (eta.exp() + 1)

    def nabla2_phi_sqrt_mul(self, theta, rhs):
        diag = 1 / (1-theta) + 1 / (1+theta)
        diag = diag.sqrt()

        return diag * rhs

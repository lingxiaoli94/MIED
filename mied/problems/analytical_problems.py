import torch
import numpy as np
import math
from abc import ABC, abstractmethod

'''
bbox corresponds to the dimension of the variable (=dim) which is
not the same as the intrinsic dimension.
'''

def sample_simplex(dim, batch_size, device):
    samples = torch.from_numpy(np.random.dirichlet(
        torch.ones([dim]) * 5, size=batch_size)).float().to(device)

    return samples

'''
Below are reparameterization options.
reparam_fn always come together with eq_fn and ineq_fn,
and prior_sample_fn.
'''

def id_reparam(dim):
    return {
        'eq_fn': None,
        'ineq_fn': None,
        'reparam_fn': None,
        'prior_sample_fn': (lambda B, device:
                            torch.randn([B, dim], device=device)),
    }


def box_id_reparam(dim):
    return {
        'eq_fn': None,
        'ineq_fn': lambda X: torch.cat([-1 - X, X - 1], -1),
        'reparam_fn': None,
        'prior_sample_fn': (lambda B, device:
                            torch.rand([B, dim], device=device) - 0.5),
    }


def box_tanh_reparam(dim):
    return {
        'eq_fn': None,
        'ineq_fn': None,
        'reparam_fn': lambda Z: torch.tanh(Z),
        'prior_sample_fn': (lambda B, device:
                            torch.atanh(torch.rand([B, dim], device=device) - 0.5)),
    }


def box_mirror_reparam(dim, entropic=False):
    if entropic:
        def nabla_psi(X):
            return torch.log(1 + X) - torch.log(1 - X)

        def nabla_psi_star(Z):
            return 1 - 2 / (Z.exp() + 1)
    else:
        def nabla_psi(X):
            return (1 / (1 - X)) - (1 / (1 + X))

        def nabla_psi_star(Z):
            return ((1 + Z**2).sqrt() - 1) / Z

    return {
        'eq_fn': None,
        'ineq_fn': None,
        'reparam_fn': lambda Z: nabla_psi_star(Z),
        'prior_sample_fn': (lambda B, device:
                            nabla_psi(torch.rand([B, dim], device=device) - 0.5)),
    }

def sphere_reparam(dim):
    return {
        'eq_fn': lambda X: X.square().sum(-1, keepdim=True) - 1,
        'ineq_fn': None,
        'reparam_fn': None,
        'prior_sample_fn': (lambda B, device:
                            torch.randn([B, dim], device=device)),
    }


def heart_id_reparam():
    return {
        'eq_fn': None,
        'ineq_fn': lambda X: ((X[:,0]**2+X[:,1]**2-1)**3-(X[:,0]**2)*(X[:,1]**3)).unsqueeze(-1),
        'reparam_fn': None,
        'prior_sample_fn': (lambda B, device:
                            torch.rand([B, 2], device=device)),
    }


def period_id_reparam():
    return {
        'eq_fn': None,
        'ineq_fn': lambda X: torch.cat([
            ((torch.cos(3 * np.pi * X[:, 0]) + torch.cos(3 * np.pi * X[:, 1])).square() - 0.3).unsqueeze(-1),
            -1-X,
            X-1], -1),
        'reparam_fn': None,
        'prior_sample_fn': (lambda B, device:
                            0.5+0.5*torch.rand([B, 2], device=device)),
    }


def simplex_id_reparam(dim):
    return {
        'eq_fn': lambda X: (X.sum(-1, keepdim=True) - 1),
        'ineq_fn': lambda X: -X,
        'reparam_fn': None,
        'prior_sample_fn': lambda B, device: sample_simplex(
            dim, B, device=device),
    }


def simplex_pos_sum_one_reparam(dim):
    return {
        'eq_fn': lambda X: (X.sum(-1, keepdim=True) - 1),
        'ineq_fn': None,
        'reparam_fn': lambda Z: Z.square(),
        'prior_sample_fn': (
            lambda b, device: sample_simplex(
                dim, b, device=device).sqrt()),
    }


def simplex_pos_sum_one_ineq_reparam(dim):
    return {
        'eq_fn': None,
        'ineq_fn': lambda X: torch.stack([X.sum(-1) - 1, 1 - X.sum(-1)], -1),
        'reparam_fn': lambda Z: Z.square(),
        'prior_sample_fn': (
            lambda B, device: sample_simplex(
                dim, B, device=device).sqrt()),
    }


def simplex_softmax_reparam(dim):
    return {
        'eq_fn': None,
        'ineq_fn': None,
        'reparam_fn': lambda Z: torch.nn.functional.softmax(Z, dim=-1),
        'prior_sample_fn': (
            lambda B, device: sample_simplex(
                dim, B, device=device).log()),
    }



# def cube_constraint(dim, bound=[0, 1]):
#     return {
#         'ineq_fn': lambda X: torch.cat([bound[0] - X, X - bound[1]], -1),
#         'bbox': torch.tensor([bound] * dim),
#         'embed_dim': dim,
#         # 'reparam_dict': {
#         #     'reparam_fn': lambda Z: torch.sigmoid(Z) * (bound[1] - bound[0]) + bound[0],
#         #     'prior_sample_fn': lambda B, device: torch.randn(
#         #         [B, dim], device=device),
#         # }
#     }


# def sphere_constraint(dim):
#     return {
#         'eq_fn': lambda X: X.square().sum(-1, keepdim=True) - 1,
#         'bbox': torch.tensor([[-1, 1]] * dim),
#         'embed_dim': dim - 1,
#     }


# def free_constraint(dim, r=3):
#     return {
#         'bbox': torch.tensor([[-r, r]] * dim),
#         'embed_dim': dim,
#         # 'prior_sample_fn': prior_sample_fn,
#     }


# def ellipse_constraint(dim):
#     assert(dim == 2)
#     return {
#         'eq_fn': lambda X: (X[..., 0].square() / 9 +
#                             X[..., 1].square() / 1).unsqueeze(-1) - 1,
#         'bbox': torch.tensor([[-4, 4], [-2, 2]]),
#         'embed_dim': dim - 1,
#     }


class DistributionBase(ABC):
    def __init__(self, *, dim, embed_dim, bbox, device):
        '''
        :param dim: ambient dimension
        '''
        self.dim = dim
        self.embed_dim = embed_dim
        self.bbox = bbox
        self.device = device
        self.gt_samples = None


    @abstractmethod
    def log_p(self, X):
        '''
        :param X: (B, D)
        :return: (B,)
        '''
        pass


    def get_reject_ineq_fn(self):
        '''
        :return: a function (B, D) -> (B, K)
        '''
        return None


    def sample_gt(self, B, refresh):
        if not refresh:
            if self.gt_samples is not None and self.gt_samples.shape[0] == B:
                return self.gt_samples

        # We sample multiple times and reject samples that don't satisfy
        # inequality constraints.

        ineq_fn = self.get_reject_ineq_fn()
        remain = B
        sample_list = []
        while remain > 0:
            samples = self.sample_gt_impl(2 * remain) # (2B, D)
            if samples is None:
                return None
            if ineq_fn is not None:
                satisfy = (ineq_fn(samples) <= 0).all(-1) # (2B)
            else:
                satisfy = torch.ones([samples.shape[0]],
                                     device=samples.device, dtype=torch.bool)
            samples = samples[satisfy, :]
            count = min(remain, samples.shape[0])
            remain -= count
            sample_list.append(samples[:count])

        self.gt_samples = torch.cat(sample_list, 0) # (B, D)
        assert(self.gt_samples.shape[0] == B)
        assert(self.gt_samples is not None)
        return self.gt_samples


    def sample_gt_impl(self, B):
        '''
        Can be overriden if it is possible to sample from the ground truth.
        :return: (B, D)
        '''
        return None


class Dirichlet(DistributionBase):
    def __init__(self, dim, *, device):
        super().__init__(dim=dim, embed_dim=dim - 1,
                         bbox=torch.tensor([[0, 1]] * dim),
                         device=device)
        alpha = np.ones([dim],
                        dtype=np.float64) * 0.1
        if alpha.shape[0] >= 3:
            alpha[:3] += np.array([90., 5., 5.])
        alpha = torch.from_numpy(alpha).float().to(device)
        self.alpha = alpha


    def log_p(self, X):
        assert(self.alpha.shape[0] == X.shape[-1])
        return ((self.alpha - 1) / 2 * (X.square() + 1e-6).log()).sum(-1)
        # return ((self.alpha - 1) * (X + 1e-40).log()).sum(-1)
        # return ((self.alpha - 1) * (X).log()).sum(-1)


    def sample_gt_impl(self, B):
        # rng = np.random.RandomState(123)
        return torch.from_numpy(np.random.dirichlet(
            self.alpha.cpu().detach(), size=B)).float().to(self.device)


class QuadraticFullDim(DistributionBase):
    def __init__(self, dim, *,
                 device,
                 ineq_fn=None,
                 seed=123):
        '''
        :param temp: smaller temp leads to smaller variance
        '''
        super().__init__(dim=dim, embed_dim=dim,
                         bbox=torch.tensor([[-2,2]] * dim, device=device),
                         device=device)

        self.ineq_fn = ineq_fn
        if seed is None:
            # Standard Gaussian
            A = np.eye(dim)
        else:
            rng = np.random.RandomState(seed)
            A_sqrt = rng.uniform(-1.0, 1.0, size=(dim, dim))
            A = A_sqrt @ A_sqrt.T
            A = np.linalg.inv(A)
            A /= np.linalg.det(A) # to have unit determinant
        self.A = torch.from_numpy(A).float().to(device)
        from torch.distributions.multivariate_normal import MultivariateNormal
        self.dist = MultivariateNormal(loc=torch.zeros([dim], device=device),
                                       covariance_matrix=self.A)


    def log_p(self, X):
        assert(self.A.shape[0] == X.shape[-1])
        return self.dist.log_prob(X)


    def sample_gt_impl(self, B):
        return self.dist.sample([B])


    def get_reject_ineq_fn(self):
        return self.ineq_fn


class StudentTFullDim(DistributionBase):
    def __init__(self, dim, *,
                 device,
                 ineq_fn=None,
                 df=2.0,
                 seed=50):
        '''
        :param temp: smaller temp leads to smaller variance
        '''
        super().__init__(dim=dim, embed_dim=dim,
                         bbox=torch.tensor([[-5, 5]] * dim, device=device),
                         device=device)

        self.ineq_fn = ineq_fn
        if seed is None:
            # Standard student T
            A = np.eye(dim)
        else:
            rng = np.random.RandomState(seed)
            A_sqrt = rng.uniform(-1.0, 1.0, size=(dim, dim))
            A = A_sqrt @ A_sqrt.T
            A = np.linalg.inv(A)
            A /= np.linalg.det(A) # to have unit determinant

        self.A = torch.from_numpy(A).float().to(device)
        self.A_inv = torch.from_numpy(np.linalg.inv(A)).float().to(device)
        from torch.distributions.studentT import StudentT
        self.dist = StudentT(df=df)

    def log_p(self, X):
        # X: (B, D), A: (D, D)
        assert(self.A.shape[0] == X.shape[-1])
        B = X.shape[0]
        Z = torch.bmm(self.A_inv.unsqueeze(0).expand(B, -1, -1), X.unsqueeze(-1)).squeeze(-1) # (B, D)
        return self.dist.log_prob(Z).sum(-1) # (B,)


    def sample_gt_impl(self, B):
        Z = self.dist.sample([B, self.dim]).to(self.device)
        X = torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1), Z.unsqueeze(-1)).squeeze(-1) # (B, D)
        return X


    def get_reject_ineq_fn(self):
        return self.ineq_fn


class UniformBox(DistributionBase):
    def __init__(self, dim, *, device):
        super().__init__(dim=dim, embed_dim=dim,
                         bbox=torch.tensor([[-1.3, 1.3]] * dim, device=device),
                         device=device)


    def log_p(self, X):
        # HACK: this will force X to be in the computation graph.
        return (X-X).sum(-1)


    def sample_gt_impl(self, B):
        return 2 * torch.rand([B, self.dim], device=self.device) - 1


class UniformHeart(DistributionBase):
    def __init__(self, *, device):
        super().__init__(dim=2, embed_dim=2,
                         bbox=torch.tensor([[-1.3, 1.3]] * 2, device=device),
                         device=device)


    def log_p(self, X):
        # HACK: this will force X to be in the computation graph.
        return (X-X).sum(-1)


    def sample_gt_impl(self, B):
        return 3 * (torch.rand([B, 2], device=self.device) - 0.5)

    def get_reject_ineq_fn(self):
        return heart_id_reparam()['ineq_fn']


class UniformSimplex3D(DistributionBase):
    def __init__(self, dim, *, device):
        assert(dim==3)
        super().__init__(dim=dim, embed_dim=dim-1,
                         bbox=torch.tensor([[-0.3, 1.3]] * dim, device=device),
                         device=device)


    def log_p(self, X):
        # HACK: this will force X to be in the computation graph.
        return (X-X).sum(-1)


    def sample_gt_impl(self, batch_size):
        A = np.array([1, 0, 0], dtype=np.float64)
        B = np.array([0, 1, 0], dtype=np.float64)
        C = np.array([0, 0, 1], dtype=np.float64)
        r1 = np.expand_dims(np.random.random([batch_size]), -1)
        r2 = np.expand_dims(np.random.random([batch_size]), -1)
        P = ((1-np.sqrt(r1)) * A + (np.sqrt(r1)*(1-r2)) * B + (r2 * np.sqrt(r1)) * C)
        return torch.from_numpy(P).to(self.device).float()


class GaussianMixtureBox(DistributionBase):
    def __init__(self, *, dim,
                 ineq_fn=None,
                 centers, variances, weights, bbox,
                 device):
        '''
        :param centers: (M, D)
        :param variances: (M,)
        :param weights: (M,)
        '''
        super().__init__(dim=dim, embed_dim=dim, bbox=bbox, device=device)
        assert(dim == centers.shape[1])
        self.centers = centers.to(device)
        self.variances = variances.to(device)
        self.weights = weights.to(device)
        self.weights /= self.weights.sum()
        self.ineq_fn = ineq_fn


    def log_p(self, X):
        '''
        :param X: (B, D)
        '''
        tmp = X.unsqueeze(1) - self.centers.unsqueeze(0) # (B, M, D)
        tmp = -tmp.square().sum(-1) / (2 * self.variances.unsqueeze(0)) # (B, M)
        coef = torch.pow(2 * np.pi * self.variances, self.dim / 2) # (M,)
        tmp = tmp.exp() / coef # (B, M)
        log_p = (tmp * self.weights).sum(-1).log()
        return log_p


    def sample_gt_impl(self, B):
        tmp = torch.randn([B, self.dim], device=self.device).unsqueeze(1) # (B, 1, D)
        tmp = tmp * self.variances.unsqueeze(-1).sqrt() # (B, M, D)
        tmp = tmp + self.centers  # (B, M, D)

        inds = torch.multinomial(self.weights, B,
                                 replacement=True).to(self.device) # (B,)

        M = self.centers.shape[0]
        D = self.dim

        flatten_idx = ((torch.arange(B, device=self.device) * M * D +
                        inds * D).unsqueeze(-1) +
                       torch.arange(D, device=self.device)) # (B, D)
        # Want: out[i, j] = tmp[i, inds[i], j]
        out = tmp.reshape(-1)[flatten_idx.reshape(-1)].reshape(B, D) # (B, D)
        return out


    def get_reject_ineq_fn(self):
        return box_id_reparam(2)['ineq_fn']


def create_problem(device, prob, reparam_name):
    from mied.problems.analytical import Analytical
    import re
    m = re.search('([0-9]+)d$', prob)
    dim_group = m.group(0)
    dim = int(dim_group[:-1])
    prob_name = prob[:-len(dim_group)-1]

    if prob_name == 'dirichlet':
        dist = Dirichlet(dim, device=device)
    elif prob_name == 'quadratic_uc':
        dist = QuadraticFullDim(dim, device=device, ineq_fn=None)
    elif prob_name == 'quadratic_2_uc':
        dist = QuadraticFullDim(dim, device=device, ineq_fn=None, seed=40)
    elif prob_name == 'std_gaussian_uc':
        dist = QuadraticFullDim(dim, device=device, ineq_fn=None, seed=None)
    elif prob_name == 'student_uc':
        dist = StudentTFullDim(dim, device=device, ineq_fn=None)
    elif prob_name == 'uniform_box':
        dist = UniformBox(dim, device=device)
    elif prob_name == 'uniform_heart':
        dist = UniformHeart(device=device)
    elif prob_name == 'uniform_simplex':
        dist = UniformSimplex3D(dim, device=device)
    elif prob_name == 'mog_box':
        assert(dim == 2)
        dist = GaussianMixtureBox(dim=2,
                                  bbox=torch.tensor([[-1.3, 1.3]]*2, device=device),
                                  centers=torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]],
                                                       dtype=torch.float32),
                                  variances=torch.tensor([0.3, 0.3, 0.3, 0.3]),
                                  weights=torch.tensor([0.25, 0.25, 0.25, 0.25]),
                                  device=device)
    elif prob_name == 'vmf':
        dist = QuadraticFullDim(dim, device=device, ineq_fn=None)
    else:
        raise Exception(f'Unknown problem name: {prob_name}')


    if reparam_name == 'id':
        reparam = id_reparam(dim)
    elif reparam_name == 'box_id':
        reparam = box_id_reparam(dim)
    elif reparam_name == 'box_tanh':
        reparam = box_tanh_reparam(dim)
    elif reparam_name == 'box_mirror':
        reparam = box_mirror_reparam(dim)
    elif reparam_name == 'box_mirror_entropic':
        reparam = box_mirror_reparam(dim, True)
    elif reparam_name == 'sphere':
        reparam = sphere_reparam(dim)
    elif reparam_name == 'heart_id':
        reparam = heart_id_reparam()
    elif reparam_name == 'period_id':
        reparam = period_id_reparam()
    elif reparam_name == 'simplex_pos_sum_one':
        reparam = simplex_pos_sum_one_reparam(dim)
    elif reparam_name == 'simplex_pos_sum_one_ineq':
        reparam = simplex_pos_sum_one_ineq_reparam(dim)
    elif reparam_name == 'simplex_id':
        reparam = simplex_id_reparam(dim)
    elif reparam_name == 'simplex_softmax':
        reparam = simplex_softmax_reparam(dim)
    else:
        raise Exception(f'Unknown reparametrization name: {reparam_name}')

    return Analytical(device=device,
                      bbox=dist.bbox,
                      in_dim=dist.dim,
                      embed_dim=dist.embed_dim,
                      log_p_fn=lambda X: dist.log_p(X),
                      gt_sample_fn=lambda B, refresh: dist.sample_gt(B, refresh),
                      **reparam)

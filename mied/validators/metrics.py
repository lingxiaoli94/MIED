'''
Metrics to evaluate samples:
1. With ground truth samples, we can compute
    a. Wasserstein distance between mu and mu^*
    b. KL(mu^* || mu)
2. With access to ground truth density (which is always the case), we can compute
    a. KSD, directly applicable
    b. KL(mu || mu^*)
'''

import torch
import numpy as np
import ot
from scipy import stats

from mied.solvers.ksdd import compute_ksd
from mied.utils.kernels import GaussianKernel
from mied.utils.batch_jacobian import compute_jacobian


def estimate_log_p(X, P):
    '''
    :param X: (B, D), samples used to build KDE
    :param P: (B, D), samples to evaluate at
    :return: density, (B,)
    '''
    kernel = stats.gaussian_kde(X.detach().cpu().numpy().T)
    P_log_p = kernel.logpdf(P.detach().cpu().numpy().T)
    return torch.from_numpy(P_log_p).to(P.device)


def filter_samples(samples, filter_range):
    if filter_range > 0:
        mask = torch.logical_and(samples < filter_range,
                                 samples > -filter_range).all(-1)
        return samples[mask]
    return samples


def batch_expected_diff_norm(X, Y, batch_size=1000):
    '''
    Compute E[||X-Y||] for energy distance.
    :param X: (N, D)
    :param Y: (M, D)
    '''

    total_size = Y.shape[0]
    cur = 0
    total = 0
    while cur < total_size:
        cur_size = min(total_size - cur, batch_size)
        tmp = X.unsqueeze(1) - Y[cur:cur+cur_size].unsqueeze(0)
        tmp = tmp.square().sum(-1).sqrt().sum()
        total += tmp.item() / X.shape[0]
        cur += cur_size
    return total / total_size


def compute_metric(source_samples, target_problem, *,
                   metric,
                   refresh,
                   gt_samples,
                   gt_multiplier,
                   ot_lib='pol',
                   ksd_sigma=1.0,
                   filter_range=-1,
                   strip_last_n=-1):
    '''
    :param source_samples: (B, D), can be on any device
    :param target_problem: an instance of ProblemBase
    :return: a scalar of the computed metric
    '''
    source_samples = filter_samples(source_samples, filter_range)
    if strip_last_n > 0:
        source_samples = source_samples[:, :-strip_last_n]
    if metric in ['sinkhorn', 'KL_st', 'chi2_st', 'energy_dist']:
        if gt_samples is None:
            target_samples = target_problem.sample_gt(
                gt_multiplier * source_samples.shape[0],
                refresh=refresh
            )
            target_samples = filter_samples(target_samples, filter_range)
            assert(target_samples is not None)
        else:
            target_samples = gt_samples
        if strip_last_n > 0:
            target_samples = target_samples[:, :-strip_last_n]

    if metric == 'sinkhorn':
        if ot_lib == 'pol':
            import ot
            source_weights = (np.ones(source_samples.shape[0]) /
                              source_samples.shape[0])
            target_weights = (np.ones(target_samples.shape[0]) /
                              target_samples.shape[0])
            M = ot.dist(source_samples.cpu().detach().numpy(),
                        target_samples.cpu().detach().numpy())
            W = ot.emd2(source_weights, target_weights, M)
            return W
        else:
            assert(ot_lib == 'geomloss')
            import geomloss
            loss = geomloss.SamplesLoss('sinkhorn', blur=0.0)
            return loss(source_samples, target_samples)

    if metric == 'energy_dist':
        # SS = (source_samples.unsqueeze(1) -
        #       source_samples.unsqueeze(0)).square().sum(-1).sqrt().mean()
        # ST = (source_samples.unsqueeze(1) -
        #       target_samples.unsqueeze(0)).square().sum(-1).sqrt().mean()
        # TT = (target_samples.unsqueeze(1) -
        #       target_samples.unsqueeze(0)).square().sum(-1).sqrt().mean()
        # return (2 * ST - SS - TT).item()
        SS = batch_expected_diff_norm(source_samples, source_samples)
        ST = batch_expected_diff_norm(source_samples, target_samples)
        TT = batch_expected_diff_norm(target_samples, target_samples)
        return (2 * ST - SS - TT)


    if metric in ['KL_ts', 'chi2_ts']:
        source_log_p = estimate_log_p(source_samples,
                                      source_samples)
        target_log_p = target_problem.eval_log_p(source_samples)
        if metric == 'KL_ts':
            return (source_log_p - target_log_p).mean().item()
        else:
            return ((target_log_p - source_log_p).exp() - 1).square().mean().item()

    if metric in ['KL_st', 'chi2_st']:
        target_log_p = target_problem.eval_log_p(target_samples)
        source_log_p = estimate_log_p(source_samples,
                                      target_samples)
        if metric == 'KL_st':
            return (target_log_p - source_log_p).mean().item()
        else:
            return ((source_log_p - target_log_p).exp() - 1).square().mean().item()

    if metric == 'KL_sym':
        return (compute_metric(source_samples, target_problem,
                              metric='KL_st') +
                compute_metric(source_samples, target_problem,
                               metric='KL_ts'))

    if metric == 'chi2_sym':
        return (compute_metric(source_samples, target_problem,
                              metric='chi2_st') +
                compute_metric(source_samples, target_problem,
                               metric='chi2_ts'))

    if metric == 'ksd':
        kernel = GaussianKernel(sigma=ksd_sigma)
        X = source_samples.detach().clone()
        X.requires_grad_(True)
        log_p = target_problem.eval_log_p(X)
        grad_log_p = compute_jacobian(log_p.unsqueeze(-1),
                                      X,
                                      create_graph=False,
                                      retain_graph=False)
        grad_log_p = grad_log_p.squeeze(-2)
        return compute_ksd(X, grad_log_p, kernel).item()

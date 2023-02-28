import torch
import argparse
from pathlib import Path
import math
import wandb

import matplotlib.pyplot as plt

from mied.validators.particle import ParticleValidator
from mied.utils.random import seed_all
from mied.utils.ec import ExperimentCoordinator
from mied.problems.analytical_problems import create_problem
from mied.solvers.mied import MIED
from mied.solvers.svgd import SVGD
from mied.solvers.ksdd import KSDD
from mied.solvers.ipd import IPD
from mied.solvers.lmc import LMC
from mied.solvers.dynamic_barrier import DynamicBarrier
from mied.solvers.no_op_projector import NoOpProjector

if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'num_itr': 2000,
        'traj_freq': 10,
        'plot_update': False,
        'num_trial': 10,
        'gt_multiplier': 10,
    })
    ec.add_common_arguments({
        'prob': 'uniform_box_2d',
        'reparam': 'box_tanh',
        'filter_range': -1,
    })
    ec.add_method_arguments(MIED, {
        'kernel': 'riesz',
        'eps': 1e-8,
        'riesz_s': -1.0,
        'alpha_mied': 0.5,
        'include_diag': 'nnd_scale',
        'diag_mul': 1.3,
    })
    ec.add_method_arguments(KSDD, {
        'sigma': 1.0,
    })
    ec.add_method_arguments(IPD, {
    })
    ec.add_method_arguments(SVGD, {
        'kernel_h': -1.0,
    })
    ec.add_method_arguments(LMC, {
        'lmc_lr': 1e-3,
        'mirror_map': 'box_entropic'
    })
    ec.add_projector_arguments(DynamicBarrier, {
        'alpha_db': 1.0,
        'merge_eq': True,
        'max_proj_itr': 20
    })
    ec.add_projector_arguments(NoOpProjector, {
    })

    ec_result = ec.parse_args()
    tmp_args = ec_result.tmp_args
    config = ec_result.config
    seed_all(config['seed'])
    problem = create_problem(ec_result.tmp_args.device,
                             config['prob'],
                             config['reparam'])
    solver = ec.create_solver(problem)
    validator = ParticleValidator(problem=problem)

    def post_step_fn(i):
        if tmp_args.traj_freq <= 0:
            return
        if (i + 1) % (tmp_args.val_freq * tmp_args.traj_freq) == 0:
            metrics = ['sinkhorn', 'energy_dist']

            result = validator.run(samples=solver.get_samples(),
                                   updates=solver.compute_update(
                                       i, solver.get_samples()),
                                   include_density=False,
                                   metrics=metrics,
                                   num_trial=tmp_args.num_trial,
                                   gt_multipler=tmp_args.gt_multiplier,
                                   filter_range=config['filter_range'],
                                   save_path=(ec_result.exp_dir /
                                              'step-{:05}.h5'.format(i + 1)))
            samples = result['samples']
            bbox = problem.bbox.cpu().detach()
            fig, ax = plt.subplots()
            ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.6)
            if tmp_args.plot_update:
                updates = result['updates']
                ax.quiver(samples[:, 0], samples[:, 1],
                          updates[:, 0], updates[:, 1],
                          angles='xy', scale_units='xy', scale=1)
            ax.set_xlim(bbox[0, :])
            ax.set_ylim(bbox[1, :])
            ax.set_aspect('equal')
            log_dict = {
                'metrics': {m: result[m] for m in metrics},
                'samples': wandb.Image(fig),
            }
            if tmp_args.num_trial > 1:
                log_dict['metrics_std'] = {m: result[m + '_std']
                                           for m in metrics}
            plt.close(fig)

            gt_samples = problem.sample_gt(
                samples.shape[0], refresh=True).cpu().detach()
            if gt_samples is not None:
                fig, ax = plt.subplots()
                ax.scatter(gt_samples[:, 0], gt_samples[:, 1], s=5)
                ax.set_xlim(bbox[0, :])
                ax.set_ylim(bbox[1, :])
                ax.set_aspect('equal')
                log_dict['gt_samples'] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(log_dict, commit=False)

    solver.run(num_itr=tmp_args.num_itr,
               post_step_fn=post_step_fn)

    validator.run(samples=solver.get_samples(),
                  include_gt=True,
                  include_density=problem.in_dim == 2,
                  density_bbox=problem.bbox,
                  save_path=ec_result.exp_dir / 'result.h5')


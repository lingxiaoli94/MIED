import torch
import argparse
from pathlib import Path
import math
import h5py
import wandb
from mied.validators.particle import ParticleValidator
from mied.utils.random import seed_all
from mied.utils.ec import ExperimentCoordinator
from mied.utils.h5_helpers import save_dict_h5
from mied.problems.fairness_bnn import FairnessBNN
from mied.solvers.mied import MIED
from mied.solvers.svgd import SVGD
from mied.solvers.ksdd import KSDD
from mied.solvers.ipd import IPD
from mied.solvers.dynamic_barrier import DynamicBarrier

if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'num_itr': 1000,
        'traj_freq': 10,
    })
    ec.add_common_arguments({
        'thres': 0.01,
        'ineq_scale': 1.0,
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
    ec.add_method_arguments(SVGD, {
    })
    ec.add_method_arguments(IPD, {
    })
    ec.add_projector_arguments(DynamicBarrier, {
        'alpha_db': 1.0,
        'merge_eq': True,
        'max_proj_itr': 20
    })

    ec_result = ec.parse_args()

    tmp_args, config = ec_result.tmp_args, ec_result.config

    seed_all(config['seed'])
    problem = FairnessBNN(device=tmp_args.device,
                          data_dir='data/',
                          thres=config['thres'],
                          ineq_scale=config['ineq_scale'])

    validator = ParticleValidator(problem=problem)
    def post_step_fn(i):
        pass

    solver = ec.create_solver(problem)
    solver.run(num_itr=tmp_args.num_itr,
               post_step_fn=post_step_fn)

    print('Validating ...')
    validator.run(samples=solver.get_samples(),
                  include_density=False,
                  save_path=ec_result.exp_dir / 'result.h5')

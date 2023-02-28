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
from mied.problems.logistics import BayesianLogistics
from mied.solvers.mied import MIED
from mied.solvers.svgd import SVGD
from mied.solvers.ksdd import KSDD
from mied.solvers.ipd import IPD
from mied.solvers.dynamic_barrier import DynamicBarrier

g_data_names = ['banana', 'breast_cancer', 'diabetis', 'flare_solar',
                'german', 'heart', 'image', 'ringnorm', 'splice',
                'thyroid', 'titanic', 'twonorm', 'waveform', 'covtype']

if __name__ == '__main__':
    root_dir = Path(__file__).resolve().parent
    ec = ExperimentCoordinator(root_dir)
    ec.add_temporary_arguments({
        'num_itr': 1000,
        'traj_freq': 10,
        'mcmc_only': False,
    })
    ec.add_common_arguments({
        'data_name': 'banana',
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
        'kernel_h': -1.0,
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

    data_name = config['data_name']
    if data_name in g_data_names:
        if data_name != 'covtype':
            data_path = 'data/benchmarks.mat'
        else:
            data_path = 'data/covertype.mat'
    else:
        raise Exception(f'Unknown dataset name: {data_name}!')

    problem = BayesianLogistics(
        device=tmp_args.device,
        data_path=data_path,
        data_name=data_name)

    # Generate ground truth using mcmc.
    (root_dir / 'mcmc').mkdir(parents=True, exist_ok=True)
    mcmc_file = root_dir / 'mcmc' / '{}.h5'.format(data_name)
    mcmc_log_file = root_dir / 'mcmc' / '{}.log'.format(data_name)

    if data_name != 'covtype':
        # MCMC for covtype is just too slow.
        if not mcmc_file.exists():
            samples = problem.mcmc(num_warmup=10000,
                                   num_sample=10000,
                                   log_file=mcmc_log_file)
            save_dict_h5({'samples': samples},
                         mcmc_file, create_dir=True)
        h5_handle = h5py.File(mcmc_file, 'r')
        mcmc_samples = torch.from_numpy(h5_handle['samples'][:]).to(problem.device)
        h5_handle.close()


    if not tmp_args.mcmc_only:
        validator = ParticleValidator(problem=problem)
        def post_step_fn(i):
            if tmp_args.traj_freq <= 0:
                return
            if (i + 1) % (tmp_args.val_freq * tmp_args.traj_freq) == 0:
                metrics = ['sinkhorn', 'energy_dist']
                samples = solver.get_samples()
                result = validator.run(samples=samples,
                                       metrics=metrics,
                                       gt_samples=mcmc_samples,
                                       strip_last_n=1,
                                       save_path=(ec_result.exp_dir /
                                                  'step-{:05}.h5'.format(i + 1)))
                log_dict = {
                    'metrics': {m: result[m] for m in metrics},
                }

                wandb.log(log_dict, commit=False)

        solver = ec.create_solver(problem)
        solver.run(num_itr=tmp_args.num_itr,
                   post_step_fn=post_step_fn if data_name != 'covtype' else None)

        print('Validating ...')
        validator.run(samples=solver.get_samples(),
                      include_density=False,
                      save_path=ec_result.exp_dir / 'result.h5')

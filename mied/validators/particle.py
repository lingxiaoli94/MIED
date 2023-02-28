import torch
import numpy as np
import math
from mied.utils.h5_helpers import save_dict_h5
from mied.utils.batch_eval import batch_eval_index
from mied.validators.metrics import compute_metric

class ParticleValidator:
    def __init__(self, *,
                 problem):
        self.problem = problem
        self.device = problem.device


    def generate_density_grid(self, *,
                              density_bbox,
                              density_grid_len=500):
        assert(density_bbox.shape[0] == 2)

        x_linspace = torch.linspace(
            density_bbox[0, 0],
            density_bbox[0, 1],
            density_grid_len, device=self.device)
        y_linspace = torch.linspace(
            density_bbox[1, 0],
            density_bbox[1, 1],
            density_grid_len, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_linspace, y_linspace, indexing='ij') # (L, L) x 2
        grid = torch.stack([grid_x, grid_y], -1) # (L, L, 2)
        grid_flat = grid.reshape(-1, 2) # (L*L, 2)

        density = batch_eval_index(
            lambda inds: self.problem.eval_log_p(
                grid_flat[inds, :]),
            grid_flat.shape[0],
            no_tqdm=True,
            batch_size=10000
        )
        density = torch.cat(density, 0) # (L*L)
        density = density.reshape(grid.shape[:2]) # (L, L)

        return {
            'density_bbox': density_bbox.detach().cpu(),
            'grid_x': grid_x.detach().cpu(),
            'grid_y': grid_y.detach().cpu(),
            'grid_density': density.detach().cpu()
        }


    def run(self, *,
            samples,
            updates=None,
            save_path=None,
            include_density=False,
            metrics=[],
            num_trial=1,
            gt_samples=None,
            gt_multiplier=10,
            filter_range=-1,
            strip_last_n=-1,
            include_gt=False,
            **kwargs):

        result_dict = {}
        result_dict.update({
            'samples': samples.detach().cpu(),
        })
        if updates is not None:
            result_dict.update({
                'updates': updates.detach().cpu()
            })

        if include_gt:
            assert(gt_samples is None)
            target_samples = self.problem.sample_gt(
                gt_multiplier * samples.shape[0],
                refresh=False
            )
            result_dict['target_samples'] = target_samples.detach().cpu()

        result_dict.update(
            self.problem.custom_eval(samples)
        )

        if include_density:
            result_dict.update(self.generate_density_grid(
                density_bbox=kwargs['density_bbox']
            ))

        for metric in metrics:
            result_list = []
            for trial in range(num_trial):
                tmp = compute_metric(samples,
                                     self.problem,
                                     metric=metric,
                                     gt_samples=gt_samples,
                                     refresh=(num_trial > 1),
                                     gt_multiplier=gt_multiplier,
                                     filter_range=filter_range,
                                     strip_last_n=strip_last_n)
                result_list.append(tmp)
            result_list = np.array(result_list)
            # if metric in ['KL_st', 'KL_ts', 'chi2_st', 'chi2_ts']:
            #     tmp = math.log(abs(tmp))
            result_dict[metric] = np.mean(result_list)
            if num_trial > 1:
                result_dict[metric + '_std'] = np.std(result_list)

        if save_path is not None:
            save_dict_h5(result_dict, save_path, create_dir=True)
        return result_dict


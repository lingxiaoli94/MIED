from abc import ABC, abstractmethod
import torch
import numpy as np
from pathlib import Path
from tqdm import trange
from mied.utils.batch_hessian import compute_hessian

class ParticleBasedSolver(ABC):
    def __init__(self, *,
                 problem,
                 projector,
                 num_particle,
                 precondition,
                 optimizer_conf,
                 direct_update=False,
                 val_freq,
                 ckpt_path,
                 logger_fn):
        '''
        Abstract base class for particle-based solvers that differ only in
        the updates.
        The default parameters are set in ExperimentCoordinator class.

        :param problem: the constrained sampling problem
        :param projector: handler of the constraints
        :param num_particle: number of particles
        '''
        self.problem = problem
        self.projector = projector
        self.precondition = precondition
        self.direct_update = direct_update
        self.val_freq = val_freq
        self.ckpt_path = ckpt_path
        self.logger_fn = logger_fn

        self.particles = self.problem.sample_prior(num_particle)
        self.particles.requires_grad_(True)

        self.optimizer_conf = optimizer_conf
        self.create_optimizer()

        self.init_global_step = 0


    def create_optimizer(self):
        conf = self.optimizer_conf
        if conf['cls'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                [self.particles], lr=conf['lr'],
                betas=(conf['beta1'], conf['beta2']),
            )
        elif conf['cls'] == 'LBFGS':
            self.optimizer = torch.optim.LBFGS(
                [self.particles], lr=conf['lr'])
        elif conf['cls'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                [self.particles], lr=conf['lr'],
                momentum=conf['beta1']
            )
        elif self.optimizer_conf['cls'] == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(
                [self.particles], lr=conf['lr'],
                alpha=conf['beta1']
            )
        else:
            raise Exception(f'Unknown optimizer class {self.optimizer_conf["cls"]}')


    def load_ckpt(self):
        '''
        :return: the current global step.
        '''
        p = Path(self.ckpt_path)
        if not p.exists():
            print('No checkpoint file found. Use default initialization.')
            self.init_global_step = 0
            return

        ckpt = torch.load(self.ckpt_path)
        global_step = ckpt['global_step']
        self.particles = ckpt['particles']
        self.particles.to(self.problem.device)
        self.particles.requires_grad_(True)
        self.create_optimizer()
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        print('Loading solver from {} at step {}...'.format(p, global_step))

        np.random.set_state(ckpt['np_rng_state'])
        torch.set_rng_state(ckpt['torch_rng_state'])
        self.init_global_step = global_step


    def save_ckpt(self, global_step):
        print('Saving solver at global step {}...'.format(global_step))
        p = Path(self.ckpt_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        all_dict = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'particles': self.particles,
            'global_step': global_step,
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state()
        }

        torch.save(all_dict, self.ckpt_path)


    @abstractmethod
    def compute_update(self, i, X):
        '''
        :param i: current step
        :param X: particles (after reparameterization)
        :return: (B, in_dim),
            * if direct_update = False, then update directions so that
                x_new = x_old + eta * x_update,
              where eta is modulated by the optimizer and the projector.
            * if direct_update = True, then x_new = x_update
        '''
        pass


    def step(self, i):
        if self.direct_update:
            # Skip using optimizer (e.g. in LMC).
            self.particles = self.compute_update(i, self.particles).detach()
            self.particles.requires_grad_(True)
            return

        self.optimizer.zero_grad()

        X = self.problem.reparametrize(self.particles)
        # Compute update w.r.t. X.
        update = self.compute_update(i, X).detach() # (B, D)

        # Optional preconditioning.
        if self.precondition:
            log_p_fn = lambda X: self.problem.eval_log_p(X)
            hess = compute_hessian(log_p_fn, X) # (B, D, D)
            update = torch.linalg.lstsq(-hess, update.unsqueeze(-1)).solution # (B, D)
            update = update.squeeze(-1)

        # The projector may modify update.
        update = self.projector.step(X,
                                     update=update,
                                     problem=self.problem)

        # Manual chain rule.
        if self.particles.grad is not None:
            self.particles.grad.zero_()
        X.backward(gradient=update, inputs=self.particles)
        update = self.particles.grad.detach()

        self.particles.grad = -update.detach()
        self.optimizer.step()


    def get_samples(self):
        '''
        Obtain the resulting samples.
        '''
        return self.problem.reparametrize(self.particles)


    def post_step(self, i):
        '''
        Stuff to do after each step, e.g., update log_msg.
        '''
        pass


    def run(self, *,
            num_itr,
            ckpt_save_freq=-1,
            post_step_fn=None):

        if ckpt_save_freq == -1:
            ckpt_save_freq = num_itr

        loop_range = trange(self.init_global_step, num_itr)
        for i in loop_range:
            self.step(i)
            if post_step_fn is not None:
                post_step_fn(i)
            self.post_step(i)
            loop_range.set_description(self.get_progress_msg())

            global_step = i + 1
            if self.ckpt_path and ckpt_save_freq:
                if global_step % ckpt_save_freq == 0:
                    self.save_ckpt(global_step)


    def post_step(self, i):
        if self.logger_fn is not None:
            if (i + 1) % self.val_freq == 0:
                log_dict = {
                    'step': i + 1,
                    'violation': self.projector.get_violation(),
                }
                log_dict.update(self.problem.custom_post_step(self.particles))
                log_dict.update(self.custom_post_step(i))
                self.logger_fn(log_dict)


    def compute_variance(self):
        samples = self.get_samples() # (B, D)
        mean = samples.mean(0) # (D,)
        dist = (samples - mean).square().sum(-1) # (B,)
        return dist.mean()


    def compute_min_separation(self):
        samples = self.get_samples() # (B, D)
        dist = (samples.unsqueeze(1) - samples.unsqueeze(0)).square().sum(-1) # (B, B)
        val, _ = torch.topk(dist, 2, largest=False, dim=-1) # (B, 2)
        return (val[:, 1].min() + 1e-8).sqrt()


    def get_progress_msg(self):
        return 'G_vio: {:6f}'.format(self.projector.get_violation())


    def custom_post_step(self, i):
        return {}

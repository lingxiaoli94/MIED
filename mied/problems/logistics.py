import torch
import torch.distributions
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import scipy.io
from mied.problems.problem_base import ProblemBase

class BayesianLogistics(ProblemBase):
    def __init__(self, *,
                 device,
                 data_path,
                 data_name='banana',
                 exp_lambda=0.01,
                 split_seed=42,
                 batch_size=50):
        self.exp_lambda = exp_lambda
        self.exp_dist = torch.distributions.Exponential(exp_lambda)

        data = scipy.io.loadmat(data_path)
        if data_name == 'covtype':
            X = torch.from_numpy(data['covtype'][:, 1:])
            Y = torch.from_numpy(data['covtype'][:, 0])
            Y[Y == 2] = 0
            self.use_batch = True
        else:
            X = torch.from_numpy(data[data_name]['x'][0][0]) # NxM
            Y = torch.from_numpy(data[data_name]['t'][0][0]) # Nx1
            Y = Y.squeeze(-1) # N
            Y[Y == -1] = 0
            self.use_batch = False
        dataset = TensorDataset(X, Y)

        N, self.M = X.shape
        N_train = int(N * 0.8)
        N_test = N - N_train
        self.train_dset, self.test_dset = torch.utils.data.random_split(
            dataset, [N_train, N_test],
            generator=torch.Generator().manual_seed(split_seed))

        # Always use batch for test.
        self.test_dl = DataLoader(self.test_dset,
                                  batch_size=batch_size,
                                  shuffle=False)

        if self.use_batch:
            self.train_dl = DataLoader(self.train_dset,
                                       batch_size=batch_size,
                                       shuffle=True)
            self.train_dl_itr = iter(self.train_dl)
        else:
            # Otherwise put everything onto device.
            self.train_X, self.train_Y = self.train_dset[:]
            self.train_X = self.train_X.to(device)
            self.train_Y = self.train_Y.to(device)

        self.dim = self.M + 1
        super().__init__(device=device,
                         in_dim=self.dim)


    def mcmc(self, num_warmup, num_sample, *,
             log_file=None):
        '''
        Generate posterior samples using MCMC to serve as ground truth.
        '''
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS


        def model(data, labels):
            alpha = numpyro.sample('alpha', dist.Exponential(self.exp_lambda))
            W = numpyro.sample('W', dist.Normal(jnp.zeros(self.M), 1.0 / alpha))
            logits = jnp.sum(W * data, axis=-1)
            return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=labels)
        data, labels = self.train_dset[:]
        data = data.numpy()
        labels = labels.numpy()
        mcmc = MCMC(NUTS(model=model), num_warmup=num_warmup,
                    num_samples=num_sample)
        mcmc.run(jax.random.PRNGKey(0), data, labels)

        from contextlib import ExitStack, redirect_stdout

        samples = mcmc.get_samples()
        W = torch.from_numpy(np.array(samples['W']))
        alpha = torch.from_numpy(np.array(samples['alpha'])).log()
        P = torch.cat([W, alpha.unsqueeze(-1)], -1)

        test_acc = self.eval_test_accurarcy(P.to(self.device))

        with ExitStack() as stack:
            if log_file is not None:
                f = stack.enter_context(open(log_file, 'w'))
                stack.enter_context(redirect_stdout(f))
            mcmc.print_summary()
            print('MCMC test accuracy: {}'.format(test_acc))

        return P


    def sample_prior(self, batch_size):
        alpha = self.exp_dist.sample([batch_size, 1]).to(self.device)
        W = torch.randn([batch_size, self.M],
                        device=self.device) / alpha.sqrt()
        return torch.cat([W, alpha.log()], -1)


    def get_embed_dim(self):
        return self.dim


    def eval_log_p(self, P):
        if self.use_batch:
            try:
                X, Y = self.train_dl_itr.next()
            except StopIteration:
                self.train_dl_itr = iter(self.train_dl)
                X, Y = self.train_dl_itr.next()
            X = X.to(self.device)
            Y = Y.to(self.device)
        else:
            X, Y = (self.train_X, self.train_Y)

        W = P[:, :-1] # BxM
        alpha = P[:, -1].exp() # B

        log_p = -(W.square().sum(-1) * alpha / 2)
        log_p = log_p - self.exp_lambda * alpha

        out_logit = (W.unsqueeze(1) * X.unsqueeze(0)).sum(-1) # BxN
        log_p_data = -F.binary_cross_entropy_with_logits(
            out_logit,
            Y.unsqueeze(0).expand(W.shape[0], -1).float(),
            reduction='none')

        log_p_data = log_p_data.sum(-1) # B
        log_p = log_p + log_p_data

        return log_p


    def eval_test_accurarcy(self, P):
        W = P[:, :-1] # BxM
        alpha = P[:, -1].exp() # B

        total_correct = 0
        total_test = 0
        for test_batch in self.test_dl:
            X, Y = test_batch
            X = X.to(self.device) # NxM
            Y = Y.to(self.device) # N

            pred = (W.unsqueeze(1) * X.unsqueeze(0)).sum(-1) # BxN
            pred = pred.sigmoid() # BxN
            # We average pred from all particles before computing accuracy.
            total_correct += (Y == (pred.mean(0) > 0.5)).sum()
            total_test += X.shape[0]
        return total_correct / total_test


    def custom_post_step(self, P):
        return {
            'train_log_p': self.eval_log_p(P).mean(),
            'test_acc': self.eval_test_accurarcy(P)
        }

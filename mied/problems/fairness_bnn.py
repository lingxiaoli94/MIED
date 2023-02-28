import torch
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import random

from mied.problems.problem_base import ProblemBase
from mied.utils.adult_loader import load_data

# Using the same setup as https://proceedings.neurips.cc/paper/2021/hash/c61aed648da48aa3893fb3eaadd88a7f-Abstract.html


class BayesianNN:
    def __init__(self, idx, X_train, y_train, batch_size, hidden_dim, thres):
        self.idx = idx
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.n_features = X_train.shape[1] - 1
        self.hidden_dim = hidden_dim
        self.thres = thres

    def forward(self, inputs, theta):
        assert(theta.shape[1] == (self.n_features + 2) * self.hidden_dim + 1)
        # Unpack theta
        w1 = theta[:, 0:self.n_features * self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features * self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)
        w2 = theta[:, (self.n_features + 1) * self.hidden_dim:(self.n_features + 2) * self.hidden_dim].unsqueeze(2)
        b2 = theta[:, -1].reshape(-1, 1, 1)

        inputs = inputs.unsqueeze(0).repeat(w1.shape[0], 1, 1)
        inter = (torch.bmm(inputs, w1) + b1).relu()
        out_logit = torch.bmm(inter, w2) + b2
        out = out_logit.squeeze()
        return out


    def get_log_prob_and_constraint(self, theta):
        model_w = theta[:, :]
        w_prior = Normal(0., 1.)

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        out_logit = self.forward(X_batch[:, self.idx], theta)  # [num_particle, batch_size]
        y_batch_repeat = y_batch.unsqueeze(0).repeat(out_logit.shape[0], 1)
        log_p_data = F.binary_cross_entropy_with_logits(out_logit, y_batch_repeat, reduction='none')
        log_p_data = (-1.)*log_p_data.sum(dim=1)

        log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)
        log_p = log_p0 + log_p_data * (self.X_train.shape[0] / self.batch_size)  # (8) in paper

        ### NOTE: compute fairness loss
        mean_sense   = X_batch[:, 45].mean()
        weight_sense = X_batch[:, 45] - mean_sense # [batch_size]
        #weight_sense = weight_sense.view(1, -1).repeat(self.num_particles, 1)
        # Modify here as well.
        out = out_logit.sigmoid()
        out = out - out.mean(dim=1, keepdim=True) # [num_particle, batch_size]
        # constrain = ((weight_sense.unsqueeze(0) * out_logit).mean(-1))**2 - self.thres
        constrain = ((weight_sense.unsqueeze(0) * out).mean(-1))**2 - self.thres

        return log_p, constrain


class FairnessBNN(ProblemBase):
    def __init__(self, data_dir,
                 thres,
                 ineq_scale,
                 device=torch.device('cpu')):
        self.ineq_scale = ineq_scale

        idx = [i for i in range(87)]
        del idx[45]

        X_train, y_train, X_test, y_test, start_index, cat_length = load_data(
            data_dir, get_categorical_info=True)
        X_train = X_train[:20000]
        y_train = y_train[:20000]
        n = X_train.shape[0]
        n = int(0.99 * n)
        # Note: X_val is not used.
        X_train = X_train[:n, :]
        y_train = y_train[:n]
        X_train = np.delete(X_train, 46, axis=1)
        X_test = np.delete(X_test, 46, axis=1)

        X_train = torch.tensor(X_train).float().to(device)
        X_test = torch.tensor(X_test).float().to(device)
        y_train = torch.tensor(y_train).float().to(device)
        y_test = torch.tensor(y_test).float().to(device)

        X_train_mean, X_train_std = torch.mean(X_train[:, idx], dim=0), torch.std(X_train[:, idx], dim=0)
        X_train[:, idx] = (X_train [:, idx]- X_train_mean) / X_train_std
        X_test[:, idx] = (X_test[:, idx] - X_train_mean) / X_train_std

        batch_size, hidden_dim = 19800, 50
        in_dim = (X_train.shape[1] - 1 + 2) * hidden_dim + 1

        super().__init__(device=device,
                         in_dim=in_dim)
        self.bnn = BayesianNN(idx,
                              X_train, y_train, batch_size, hidden_dim, thres)

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.idx = idx


    def sample_prior(self, batch_size):
        return 0.1 * torch.randn([batch_size, self.in_dim], device=self.device)


    def eval_log_p(self, theta):
        log_p, constraint = self.bnn.get_log_prob_and_constraint(theta)
        return log_p


    def eval_ineq(self, theta):
        log_p, constraint = self.bnn.get_log_prob_and_constraint(theta)
        return self.ineq_scale * constraint.unsqueeze(-1)


    def get_embed_dim(self):
        return self.in_dim  # full dimension


    def custom_eval(self, theta):
        X_test = self.X_test
        y_test = self.y_test
        with torch.no_grad():
            prob = self.bnn.forward(X_test[:, self.idx], theta)
            y_pred = torch.sigmoid(prob).mean(dim=0)  # Average among outputs from different network parameters(particles)
            y_pred = y_pred.cpu().numpy()
            sum_positive = np.zeros(2).astype(float)
            count_group = np.zeros(2).astype(float)
            for j in range(2):
                A = y_pred[X_test.cpu().numpy()[:,45]==j]
                count_group[j] = A.shape[0]
                sum_positive[j] = np.sum(A >= 0.5)
            ratio = sum_positive/count_group
            CV = np.max(ratio) - np.min(ratio)

            y_pred[y_pred>= 0.5] = 1
            y_pred[y_pred<0.5] = 0
            acc_bnn = np.sum(y_pred==y_test.cpu().numpy())/float(y_test.shape[0])
            cv_bnn = CV
            print('acc: ', np.sum(y_pred==y_test.cpu().numpy())/float(y_test.shape[0]), 'fairness:', CV)

            acc_cllt = []
            cv_cllt = []
            for i in range(prob.shape[0]):
                y_pred = torch.sigmoid(prob[i, :])
                y_pred = y_pred.cpu().numpy()
                sum_positive = np.zeros(2).astype(float)
                count_group = np.zeros(2).astype(float)
                for j in range(2):
                    A = y_pred[X_test.cpu().numpy()[:,45]==j]
                    count_group[j] = A.shape[0]
                    sum_positive[j] = np.sum(A >= 0.5)
                ratio = sum_positive/count_group
                CV = np.max(ratio) - np.min(ratio)

                y_pred[y_pred>= 0.5] = 1
                y_pred[y_pred<0.5] = 0
                acc_cllt.append(np.sum(y_pred==y_test.cpu().numpy())/float(y_test.shape[0]))
                cv_cllt.append(CV)
            # print('mean CV {}, best CV {}, worst CV {}'.format(
            #       np.mean(np.array(cv_cllt)),
            #       np.min(np.array(cv_cllt)),
            #       np.max(np.array(cv_cllt))))

            return {
                'acc_all': np.stack(acc_cllt, 0),
                'cv_all': np.stack(cv_cllt, 0),
                'acc_bnn': acc_bnn,
                'cv_bnn': cv_bnn,
            }


    def custom_post_step(self, theta):
        eval_dict = self.custom_eval(theta)
        del eval_dict['acc_all']
        del eval_dict['cv_all']

        log_p, constraint = self.bnn.get_log_prob_and_constraint(theta)
        constraint = constraint + self.bnn.thres
        # Average across all particles.
        eval_dict['log_p'] = log_p.sum(-1).mean()
        eval_dict['constraint_mean'] = constraint.mean()
        eval_dict['constraint_max'] = constraint.max()
        return eval_dict

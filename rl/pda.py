""" Policy dual averaging """
from abc import abstractmethod

import warnings 

import numpy as np
import numpy.linalg as la

import torch

import gymnasium as gym

from rl.pmd import FOPO
from rl import Rollout
from rl import NNFunctionApproximator
from rl.utils import vec_to_int
from rl.utils import safe_normalize_row
from rl.utils import rbf_kernel
from rl.utils import get_rbf_features
from rl.utils import get_space_property
from rl.utils import get_space_cardinality
from rl.utils import pretty_print_gridworld
from rl.utils import RunningStat

from rl.gopt import ACFastGradDescent
from rl.gopt import BlackBox

class PDAGeneralStateAction(FOPO):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.check_PDAGeneralStateAction_params()

        self.action_space = env.action_space
        self.obs_space = env.observation_space
        (action_is_finite, action_dim, _) = get_space_property(self.action_space)
        (obs_is_finite, obs_dim, _) = get_space_property(self.obs_space)

        if isinstance(obs_dim, tuple):
            assert len(obs_dim) == 1, "Can only handle 1D observation space"
            self.obs_dim = obs_dim[0]

        if isinstance(action_dim, tuple):
            assert len(action_dim) == 1, "Can only handle 1D action space"
            self.action_dim = action_dim[0]
            
        if(action_is_finite or obs_is_finite):
            raise Exception("Both action and state space must be finite")

        self.obs_runstat = RunningStat(obs_dim)
        self.action_runstat = RunningStat(action_dim)
        self.rwd_runstat = RunningStat(1)
        self.updated_at_least_once = False
        # TODO: Make this customizaible? Or just set to all zeros
        self._last_a = self.pi_0 = np.zeros(action_dim, dtype=float)
        self.pi_0_tensor = torch.from_numpy(self.pi_0)

        self.mu_Q = 0
        self.mu_d = params["mu_h"] - self.mu_Q
        self.prev_beta_sum = 0
        self.beta_sum = 0
        self.sampling_grad = []

        # uniform policy and function approximation
        self.env_warmup()
        (_, _, s_visited, a_visited) = self.rollout.get_est_stateaction_value()
        sa_dim = self.obs_dim+self.action_dim
        # This is for Q^k_{[k]} and Q(\pi_k)
        self.fa_Q_accum = NNFunctionApproximator(1, input_dim=sa_dim, output_dim=1, params=params)
        self.fa_Q = NNFunctionApproximator(1, input_dim=sa_dim, output_dim=1, params=params)

    def check_PDAGeneralStateAction_params(self):
        self.params["mu_h"] = self.params.get("mu_h", 0)

    def get_stepsize_schedule(self):
        """ Returns (beta_t, lambda_t) """
        if self.mu_d > 0:
            return (self.t+1, self.mu_d)
        elif self.mu_d == 0:
            base_stepsize = self.params.get("base_stepsize", 1.)
            # TODO: How to make this parameter free
            lam = base_stepsize
            return (self.t+1, lam*(self.t+1)**(1.5))
        else:
            # if self.params["stepsize"] == "decreasing":
            #     return (self.t+1, (self.t+1)*abs(self.mu_d))
            max_iter = self.params["max_iter"]
            return (self.t+1, max_iter*(max_iter+1)*abs(self.mu_d))

    def policy_sample(self, s):
        """ 
        Sample via AC-FGM 

        We sample according to
        \[
            \min_{a \in A} { \sum_{t=0}^k \beta_t Q(s,a;theta_t) + \lambda_k D(\pi_0(s),a) }
        \]
        by scaling down by $(\sum_{t=0}^k \beta_t)^{-1}$ and solving the equivalent problem
        \[
            \min_{a \in A} { Q^k(s,a;theta_{[k]}) + (\lambda_k/\sum_{t=0}^k \beta_t) D(\pi_0(s),a) }
        \]
        where $Q^k(s,a;theta_{[k]}) := (\sum_{t=0}^k \beta_t)^{-1} \sum_{t=0}^k \beta_t Q(s,a;\theta_t)$.
        So we need to ensure the tolerance is also scaled by down by $(\sum_{t=0}^k \beta_t)^{-1}$.
        """
        if not self.updated_at_least_once:
            return self.pi_0

        s_ = np.copy(s)
        warm_start = True
        if self.just_updated_policy:
            self.sampling_grad = []
            self.mu_Q = 0.
            self.mu_d = self.params["mu_h"]

            # s_diff = la.norm(self._last_s - s)
            warm_start = False

            """
            # guess the size of the gradient of NN wrt a
            df = lambda a : -self.fa_Q_accum.grad(np.atleast_2d(np.append(s_, a)))[len(s_):] 
            m = 32
            df_norm_arr = np.zeros(m, dtype=float)
            for i in range(m):
                a = self.env.action_space.sample()
                df_norm_arr[i] = la.norm(df(a))
            est_Q_grad_norm = np.mean(df_norm_arr) + 2*np.std(df_norm_arr) 
            self.scale = min(1, 1./est_Q_grad_norm)
            """

            self.just_updated_policy = False
            self._first_eta = -1

        # TODO: How to automate this?
        a_0 = self._last_a if warm_start else self.pi_0

        while 1:
            (_, lam_t) = self.get_stepsize_schedule()

            scale = lam_t/(self.beta_sum)

            # TODO: Add regularization
            f = lambda a : -self.fa_Q_accum.predict(np.atleast_2d(np.append(s_, a)))  \
                        + scale*0.5*la.norm(a-self.pi_0)**2
            df = lambda a : -self.fa_Q_accum.grad(np.atleast_2d(np.append(s_, a)))[len(s_):]  \
                        + scale*(a-self.pi_0)
            oracle = BlackBox(f, df)

            # stop_nonconvex = abs(self.mu_d) <= ub_est_Q_grad_norm
            stop_nonconvex = True
            opt = ACFastGradDescent(
                oracle, 
                np.copy(self.pi_0), # a_0, 
                alpha=0.0, 
                tol=1e-4/self.beta_sum, 
                stop_nonconvex=stop_nonconvex,
                first_eta=self._first_eta
            )

            max_iter = 1_000 if warm_start else 50_000

            a_hat, _, grad_hist = opt.solve(n_iter=max_iter)
            self._first_eta = opt.first_eta
            self.sampling_grad.append(grad_hist[-1])

            self._last_a = np.copy(a_hat)

            if stop_nonconvex and opt.detected_nonconvex:
                old_mu_d = self.mu_d
                self.mu_Q = 2*self.mu_Q if self.mu_Q > 0 else min(1, scale**2)
                self.mu_d = self.params["mu_h"] - self.mu_Q
            else:
                break

        return a_hat

    def policy_evaluate(self):
        """ Estimates Q function and stores in self.Q_est """
        # TODO: Add more options
        self.monte_carlo_Q()

    def monte_carlo_Q(self):
        """ 
        Constructs Q/advantage function by Monte-Carlo. If missing any data
        points, we use RKHS to fill missing points.

        We update our running stats (e.g., mean and variance) for next
        iteration
        """
        # check if we are missing any points 
        (q_est, adv_est, s_visited, a_visited) = self.rollout.get_est_stateaction_value(
            self._last_pred_value,
            self._last_state_done,
        )

        X = np.hstack((s_visited, a_visited))
        y = adv_est if self.params.get("use_advantage", False) else q_est
        if self.params.get("normalize_sa_val", False):
            y /= (np.std(y)+1e-8)

        self._last_sa_visited = np.copy(X)
        self._last_y = np.copy(y)

        self.last_pe_loss, _ = self.fa_Q.update(X, y, validation_frac=0)

    def policy_update(self): 
        self.updated_at_least_once = True
        self.l2_policy_update()

    def l2_policy_update(self):
        """ 
        Policy update with PMD and Euclidean distance. Solve

        min_\theta{ | Q^k(s,a;\theta) - \bar{\beta}_k^{-1}*[\bar{\beta}_{k-1}*Q^{k-1}(s,a; \theta_{[k-1]}) + Q(s,a; \theta_k) |_2}

        """
        self.just_updated_policy = True

        (beta_t, _) = self.get_stepsize_schedule()
        self.prev_beta_sum = self.beta_sum
        self.beta_sum += beta_t

        X = self._last_sa_visited
        y_arr = []
        # iterate over samples we used to estimate the last Q_{\pi_k}
        Q_acc_pred = self.fa_Q_accum.predict(X)
        y_i        = self._last_y
        y = (self.beta_sum)**(-1)*(self.prev_beta_sum*Q_acc_pred + beta_t*y_i)

        # approximately solve apolicy update
        self.last_po_loss, _ = self.fa_Q_accum.update(X,y, validation_frac=0)

    def policy_performance(self) -> float: 
        """ Uses policy estimation from `policy_evaluate()` and updates new policy (can
            differ depending on policy approximation).
        """
        # TODO: Implement this
        return 0

    def get_value_function(self):
        raise NotImplemented

    def env_warmup(self):
        """ 
        Runs the environment for a fixed number of iterations estimate
        empirical mean and variance of observations, actions, and rewards
        """
        old_rollout_len = self.params["rollout_len"]
        self.params["rollout_len"] = max(2048, old_rollout_len)
        self.collect_rollouts()
        self.params["rollout_len"] =  old_rollout_len

        # TODO: Can we delete these?
        # self.obs_runstat.update()
        # self.action_runstat.update()
        self.rwd_runstat.update()

        print("Finished normalization warmup")

    def normalize_obs(self, s):
        self.obs_runstat.push(s)
        if self.params.get("normalize_obs", False):
            return np.divide(s-self.obs_runstat.mean, np.sqrt(self.obs_runstat.var))
        return s

    def normalize_action(self, a):
        self.action_runstat.push(a)
        if self.params.get("normalize_action", False):
            return np.divide(a-self.action_runstat.mean, np.sqrt(self.action_runstat.var))
        return a

    def normalize_rwd(self, r):
        """ Only scale reward, do not re-center """
        self.rwd_runstat.push(r)
        if self.params.get("normalize_rwd", False):
            return np.divide(r, np.sqrt(self.rwd_runstat.var))
            # return np.divide(r-self.rwd_runstat.mean, np.sqrt(self.rwd_runstat.var**0.5))
        return r

    def estimate_value(self, s):
        if not self.updated_at_least_once:
            return 0

        pi_k_s = self.policy_sample(s)
        v_s = self.fa_Q.predict(np.atleast_2d(np.append(s, pi_k_s)))

        return v_s

    def prepare_log(self):
        l = 15
        super().prepare_log()
        self.msg += "train/\n"
        self.msg += f"  {'pe_loss':<{l}}: {self.last_pe_loss:.4e}\n"
        self.msg += f"  {'po_loss':<{l}}: {self.last_po_loss:.4e}\n"
        self.msg += f"  {'sampl_grad_mean':<{l}}: {np.mean(self.sampling_grad):.4e}\n"
        self.msg += f"  {'sampl_grad_std':<{l}}: {np.std(self.sampling_grad):.4e}\n"
        self.msg += f"  {'mu_d':<{l}}: {self.mu_d:.4e}\n"

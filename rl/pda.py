""" Policy dual averaging """
from abc import abstractmethod

import warnings 
import os

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

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
from rl.utils import safe_mean
from rl.utils import safe_std

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
        self.projection = lambda a : a
        if self.params['pda_subprob_proj']:
            self.projection = lambda a : np.minimum(
                self.env.action_space.high,
                np.maximum(self.env.action_space.low, a)
            )

        self.mu_Q = 0
        self.mu_d = abs(params["pmd_mu_h"] - self.mu_Q)
        self.sampling_grad = []
        self.has_warned_about_nonconvex = False

        # uniform policy and function approximation
        self.env_warmup()
        (_, _, s_visited, a_visited) = self.rollout.get_est_stateaction_value()
        sa_dim = self.obs_dim+self.action_dim
        # This is for Q^k_{[k]} and Q(\pi_k), respectively
        self.fa_Q_accum = NNFunctionApproximator(
            input_dim=sa_dim, 
            output_dim=1, 
            params=params
        )
        self.fa_Q = NNFunctionApproximator(
            input_dim=sa_dim, 
            output_dim=1, 
            params=params
        )

    def check_PDAGeneralStateAction_params(self):
        assert self.params["pmd_stepsize_type"] in ['pda_1', 'pda_2'], "PDA can only do stepsize_type pda_1 or pda_2"

    def policy_sample(self, s):
        """ 
        Sample via AC-FGM 

        We sample according to
        $$
            min_{a in A} { sum_{t=0}^k beta_t Q(s,a;theta_t) + lambda_k D(pi_0(s),a) }
        $$

        by scaling down by $(sum_{t=0}^k beta_t)^{-1}$ and solving the equivalent problem

        $$
            min_{a in A} { Q^k(s,a;theta_{[k]}) + (lambda_k/sum_{t=0}^k beta_t) D(pi_0(s),a) }
        $$

        where $Q^k(s,a;theta_{[k]}) := (sum_{t=0}^k beta_t)^{-1} sum_{t=0}^k beta_t Q(s,a;theta_t)$.
        And recall the neural network fa_Q_acc learnes $Q^k(s,a;theta_{[k]})$, 
        so no additional scaling is needed.

        So we need to ensure the tolerance is also scaled by down by 
        $(sum_{t=0}^k beta_t)^{-1}$.
        """

        t = max(1, self.t+1 if hasattr(self, 't') else 1)
        scale = np.sqrt(float(self.params['pda_policy_noise'])/t)
        scale = max(scale, self.params['pda_policy_min_noise'])

        if not self.updated_at_least_once:
            # return self.projection(self.pi_0)
            return self.projection(self.pi_0 + np.random.normal(
                loc=0.0,
                scale=scale,
                size=len(self.pi_0), 
            ))

        s_ = np.copy(s)
        warm_start = True
        just_updated_policy = self.just_updated_policy
        if self.just_updated_policy:
            self.sampling_grad = []
            self.mu_Q = 0.
            self.mu_d = self.params["pmd_mu_h"]

            # s_diff = la.norm(self._last_s - s)
            warm_start = True

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

            tol_scale = 1./(self.curr_beta_sum)
            bregman_scale = lam_t * tol_scale

            # TODO: Add regularization
            # We take the negative since we want to maximize
            idxs = np.array([[0]])
            f = lambda a : -self.fa_Q_accum.predict(np.atleast_2d(np.append(s_, a)), idxs)  \
                        + bregman_scale*0.5*la.norm(a-self.pi_0)**2
            df = lambda a : -self.fa_Q_accum.grad(np.atleast_2d(np.append(s_, a)), idxs)[len(s_):]  \
                        + bregman_scale*(a-self.pi_0)
            # xs = np.linspace(-3,3,100)
            # plt.plot(xs, [f(x) for x in xs])
            # plt.title("bregman scale=%.2e" % bregman_scale)
            # plt.show()
            oracle = BlackBox(f, df)

            # stop_nonconvex = abs(self.mu_d) <= ub_est_Q_grad_norm
            opt = ACFastGradDescent(
                oracle, 
                np.copy(self.pi_0), # a_0, 
                self.projection,
                alpha=0.0, 
                tol=1e-3 * tol_scale,
                stop_nonconvex=self.params['pda_stop_nonconvex'],
                first_eta=self._first_eta
            )

            max_iter = 10
            if just_updated_policy or self._last_state_done:
                max_iter = 100
            a_hat, f_hist, grad_hist = opt.solve(n_iter=max_iter)

            if self.params['pda_plot_f'] and just_updated_policy:
                f_0 = lambda a : -self.fa_Q_accum.predict(np.atleast_2d(np.append(np.zeros(len(s)), a)), idxs)  \
                            + bregman_scale*0.5*la.norm(a-self.pi_0)**2
                df_0 = lambda a : -self.fa_Q_accum.grad(np.atleast_2d(np.append(np.zeros(len(s)), a)), idxs)[len(s_):]  \
                            + bregman_scale*(a-self.pi_0)

                plt.style.use('ggplot')
                fig, axes = plt.subplots(ncols=2,nrows=2)
                xs = np.linspace(-3,3,1000,endpoint=True)
                axes[0,0].plot(xs, [f_0(x) for x in xs])
                axes[0,1].plot(xs, [la.norm(df_0(x)) for x in xs])
                axes[0,0].set(
                    xlabel="a",
                    ylabel='f(a)',
                )
                axes[0,0].set_title("f at 0", fontsize=8)
                axes[0,1].set(
                    xlabel="a",
                    ylabel=r"$\nabla f(a)$",
                )

                axes[1,0].plot(f_hist)
                axes[1,1].plot(grad_hist)
                axes[1,0].set(title="Convergence of f")
                axes[1,1].set(title="Convergence of gradient", yscale='log')
                plt.tight_layout()

                pic_name = os.path.join("plots", "iter=%i.png" % self.t)
                plt.savefig(pic_name)
                # plt.show()
                plt.close()
            
            self._first_eta = opt.first_eta
            self.sampling_grad.append(grad_hist[-1])

            self._last_a = np.copy(a_hat)

            if self.params['pda_stop_nonconvex'] and opt.detected_nonconvex:
                if self.params['pmd_stepsize_type'] == 'pda_1' and not self.has_warned_about_nonconvex:
                    warnings.warn('PDA detected nonconvexity but is using convex stepsize sequence pda_1 instead of nonconvex sequence pda_2')
                    self.has_warned_about_nonconvex = True
                old_mu_d = self.mu_d
                self.mu_Q = 2*self.mu_Q if self.mu_Q > 0 else 1
                self.mu_d = self.params["pmd_mu_h"] - self.mu_Q
            else:
                break

        a_hat_noisy = self.projection(a_hat + np.random.normal(
            loc=0.,
            scale=scale,
            size=len(self.pi_0), 
        ))

        return a_hat_noisy

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

        X = np.hstack((
            np.array([self.normalize_obs(s) for s in s_visited]), 
            a_visited
        ))
        y = adv_est if self.params["pmd_use_adv"] else q_est
        if self.params["pmd_normalize_sa_val"]:
            y /= (np.std(y)+1e-8)

        self._last_sa_visited = np.copy(X)
        self._last_y = np.copy(y)

        idxs = np.zeros(shape=(len(y),1))
        train_losses, test_losses = self.fa_Q.update(X, idxs, y, validation_frac=0)

        self.last_pe_loss = -1 if len(train_losses) == 0 else np.mean(train_losses)
        self.last_pe_test_loss = -1 if len(test_losses) == 0 else np.mean(test_losses)

    def policy_update(self): 
        self.updated_at_least_once = True
        self.just_updated_policy = True
        self.l2_policy_update()

    def l2_policy_update(self):
        """ 
        Policy update with PMD and Euclidean distance. Solve

        min_theta{ | Q^k(s,a;theta) - bar{beta}_k^{-1}*[bar{beta}_{k-1}*Q^{k-1}(s,a; theta_{[k-1]}) + Q(s,a; theta_k) |_2}

        """
        (beta_t, lam_t) = self.get_stepsize_schedule()
        mu_h = self.params["pmd_mu_h"]
        alpha_1 = self.prev_beta_sum/self.curr_beta_sum
        alpha_2 = beta_t/self.curr_beta_sum

        y = self._last_y
        idxs = np.zeros(shape=(len(self._last_sa_visited), 1))
        Q_acc_pred = self.fa_Q_accum.predict(self._last_sa_visited, idxs)
        target = alpha_1 * Q_acc_pred + alpha_2*y

        # approximately solve apolicy update
        train_losses, test_losses = self.fa_Q_accum.update(
            self._last_sa_visited,
            idxs,
            target,
            validation_frac=0,
        )

        self.last_po_loss = -1 if len(train_losses) == 0 else np.mean(train_losses)
        self.last_po_test_loss = -1 if len(test_losses) == 0 else np.mean(test_losses)

        return train_losses, test_losses

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
        old_rollout_len = self.params["pmd_rollout_len"]
        run_rollout = self.params["pmd_normalize_obs"] or self.params["pmd_normalize_rwd"] 
        if run_rollout:
            self.params["pmd_rollout_len"] = max(2048, old_rollout_len)
        else:
            self.params["pmd_rollout_len"] = 1
        self.collect_rollouts()
        self.params["pmd_rollout_len"] =  old_rollout_len

        (q_est, adv_est, s_visited, a_visited) = self.rollout.get_est_stateaction_value()
        [self.normalize_obs(s) for s in s_visited]
        y = adv_est if self.params["pmd_use_adv"] else q_est
        [self.normalize_rwd(y_i) for y_i in y]

        # TODO: Can we delete these?
        # self.obs_runstat.update()
        # self.action_runstat.update()
        self.rwd_runstat.update()

        print("Finished normalization warmup")

    def normalize_obs(self, s):
        self.obs_runstat.push(s)
        if self.params["pmd_normalize_obs"]:
            return np.divide(s-self.obs_runstat.mean, np.sqrt(self.obs_runstat.var))
        return s

    def normalize_action(self, a):
        self.action_runstat.push(a)
        if self.params["pmd_normalize_action"]:
            return np.divide(a-self.action_runstat.mean, np.sqrt(self.action_runstat.var))
        return a

    def normalize_rwd(self, r):
        """ Only scale reward, do not re-center """
        self.rwd_runstat.push(r)
        if self.params["pmd_normalize_rwd"]:
            return np.divide(r, np.sqrt(self.rwd_runstat.var))
            # return np.divide(r-self.rwd_runstat.mean, np.sqrt(self.rwd_runstat.var**0.5))
        return r

    def estimate_value(self, s):
        if not self.updated_at_least_once:
            return 0

        pi_k_s = self.policy_sample(s)
        idxs = np.array([[0]])
        v_s = self.fa_Q.predict(np.atleast_2d(np.append(s, pi_k_s)), idxs)

        return v_s

    def prepare_log(self):
        l = 15
        super().prepare_log()
        self.msg += "train/\n"
        self.msg += f"  {'pe_loss':<{l}}: {self.last_pe_loss:.4e}\n"
        self.msg += f"  {'po_loss':<{l}}: {self.last_po_loss:.4e}\n"
        self.msg += f"  {'sampl_grad_mean':<{l}}: {safe_mean(self.sampling_grad):.4e}\n"
        self.msg += f"  {'sampl_grad_std':<{l}}: {safe_std(self.sampling_grad):.4e}\n"
        self.msg += f"  {'mu_d':<{l}}: {self.mu_d:.4e}\n"

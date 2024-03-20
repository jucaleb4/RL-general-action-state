""" Policy mirror descent """
from abc import abstractmethod

import warnings 
import time

import numpy as np
import numpy.linalg as la

import gymnasium as gym

from rl import RLAlg
from rl import Rollout
from rl import LinearFunctionApproximator
from rl.utils import vec_to_int
from rl.utils import safe_normalize_row
from rl.utils import rbf_kernel
from rl.utils import get_rbf_features
from rl.utils import get_space_property
from rl.utils import get_space_cardinality
from rl.utils import pretty_print_gridworld
from rl.utils import RunningStat

class FOPO(RLAlg):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.check_params()

        # initialize
        self.t = 0
        self._last_s, _ = env.reset()

        self.rollout = Rollout(env, **params)
        self.episode_rewards = []

        self.n_ep = 0
        self.n_step = 0
        self.last_iter_ep = 0
        self.max_ep = params["max_ep"] if params["max_ep"] > 0 else np.inf

    def _learn(self, max_iter):
        """ Runs PMD algorithm for `max_iter`s """
        checkpoint = np.linspace(0, max_iter, num=10, dtype=int)
        self.s_time = time.time()

        for t in range(max_iter):
            self.t = t

            self.params["eps_explore"] = 0. # 0.05 + 0.95 * 0.99**t

            self.collect_rollouts()

            if self.n_ep == self.max_ep:
                break

            self.policy_evaluate()

            self.policy_update()

            self.prepare_log()
            self.dump_log()

            if t in checkpoint:
                self.save_episode_reward_and_len()

        self.save_episode_reward_and_len()
        # return moving average reward
        ep_cum_rwds = self.rollout.get_ep_rwds()
        t = min(25, len(ep_cum_rwds))
        return np.mean(ep_cum_rwds[-t:])

    def check_params(self):
        if "rollout_len" not in self.params:
            warnings.warn("Did not pass in 'rollout_len' into params, defaulting to 1000")
            self.params["rollout_len"] = 1000

    def collect_rollouts(self):
        """ Collect samples for policy evaluation """
        self.rollout.clear_batch()

        s = self._last_s

        if self.params["max_ep_per_iter"] > 0:
            max_num_resets = self.params["max_ep_per_iter"]  
        else:
            max_num_resets = np.inf
        num_resets = 0
        for t in range(self.params["rollout_len"]): 
            v_s = self.estimate_value(s)

            a = self.policy_sample(s)
            (next_s, r_raw, term, trunc, _)  = self.env.step(a)
            r = self.normalize_rwd(r_raw)
            # if truncated due to time limit, bootstrap cost-to-go
            if trunc:
                v_next_s = self.estimate_value(next_s)
                r += self.params["gamma"] * v_next_s
            if t == self.params["rollout_len"]-1:
                self._last_pred_value = self.estimate_value(next_s)
                self._last_state_done = done
            done = term or trunc
            self.rollout.add_step_data(s, a, r, done, v_s, r_raw=r_raw)
            s = next_s
            self.n_step += 1
            if done:
                self.n_ep += 1
                if self.n_ep == self.max_ep:
                    return 
                s, _ = self.env.reset()
                self._last_s = np.copy(s)
                num_resets += 1
                if num_resets == max_num_resets:
                    self._last_pred_value = self.estimate_value(next_s)
                    self._last_state_done = True
                    break

    @abstractmethod
    def policy_evaluate(self):
        """ Uses samples to estimate policy value function """
        raise NotImplemented

    @abstractmethod
    def policy_update(self): 
        """ Uses policy estimation from `policy_evaluate()` and updates new policy (can
            differ depending on policy approximation).
        """
        raise NotImplemented

    @abstractmethod
    def policy_performance(self) -> float: 
        """ Uses policy estimation from `policy_evaluate()` and updates new policy (can
            differ depending on policy approximation).
        """
        raise NotImplemented

    def get_stepsize_schedule(self):
        # no strong regularization
        mu_h = self.params.get("mu_h", 0)
        if mu_h == 0:
            eta_0 = max(0.01, np.sqrt(1-self.params["gamma"]))
            base_stepsize = self.params.get("base_stepsize", eta_0)
            if base_stepsize <= 0:
                base_stepsize = eta_0
            if self.params.get("stepsize", "constant") == "constant":
                return base_stepsize
            if self.params["stepsize"] == "decreasing":
                return base_stepsize * float(self.t+1)**(-0.5)
        else:
            # base_stepsize = self.params.get("base_stepsize", 1.)
            # if base_stepsize <= 0:
            #     base_stepsize = 1.
            # if self.params["dynamic_stepsize"]:
            #     base_stepsize /= (2*scale + mu_h)
            # if self.params.get("stepsize", "constant") == "constant":
            #     return base_stepsize
            # if self.params["stepsize"] == "decreasing":
            #     return base_stepsize * float(self.t+1)**(-1)
            base_stepsize = (mu_h * self.t+1)**(-1)
            stepsize = (mu_h + base_stepsize)**(-1)
            stepsize *= (mu_h+1)**(self.t+1)

    def env_warmup(self):
        return

    def normalize_obs(self, s):
        return s

    def normalize_action(self, a):
        return a

    def normalize_rwd(self, r):
        return r

    def estimate_value(self, s):
        raise NotImplemented

    def prepare_log(self):
        l = 15
        self.msg = "-"*30 + "\n"

        rwd_arr = self.rollout.get_ep_rwds()
        len_arr = self.rollout.get_ep_lens()
        rwd_arr_trunc = rwd_arr[self.last_iter_ep: self.n_ep]
        len_arr_trunc = len_arr[self.last_iter_ep: self.n_ep]
        i = min(25, len(rwd_arr))
        moving_avg = np.mean(rwd_arr[-i:])


        self.msg += "rollout/\n"
        self.msg += f"  {'itr_ep_len_mean':<{l}}: {np.mean(len_arr_trunc):.2f}\n"
        self.msg += f"  {'itr_ep_rwd_mean':<{l}}: {np.mean(rwd_arr_trunc):.2f}\n"
        self.msg += f"  {'itr_n_ep':<{l}}: {int(self.n_ep-self.last_iter_ep)}\n"
        self.msg += f"  {'ep_rwd_25-ma':<{l}}: {moving_avg:.2f}\n"

        self.msg += "time/\n"
        self.msg += f"  {'itr':<{l}}: {self.t}\n"
        self.msg += f"  {'time_elap':<{l}}: {time.time()-self.s_time:.2f}\n"
        self.msg += f"  {'tot_steps':<{l}}: {self.n_step}\n"
        self.msg += f"  {'tot_ep':<{l}}: {self.n_ep}\n"

        self.last_iter_ep = self.n_ep

    def dump_log(self):
        self.msg += "-"*30 + "\n"
        print(self.msg, end="")
        self.msg = ""

    def save_episode_reward_and_len(self):
        rwd_arr = self.rollout.get_ep_rwds()
        len_arr = self.rollout.get_ep_lens()

        if len(self.params.get("fname", "")) == 0:
            warnings.warn("No filename given, not saving")
            return
        fmt="%1.2f,%i"
        arr = np.vstack((np.atleast_2d(rwd_arr), np.atleast_2d(len_arr))).T
        with open(self.params["fname"], "wb") as fp:
            fp.write(b"episode rewards,episode len\n")
            np.savetxt(fp, arr, fmt=fmt)
        print(f"Saved episode data to {self.params['fname']}")

class PMDFiniteStateAction(FOPO):
    def __init__(self, env, params):
        super().__init__(env, params)

        self.action_space = env.action_space
        self.obs_space = env.observation_space
        (action_is_finite, _, _) = get_space_property(self.action_space)
        (obs_is_finite, _, _)  = get_space_property(self.obs_space)

        assert action_is_finite, "Action space not finite"
        assert obs_is_finite, "Observation space not finite"

        # uniform policy
        self.n_states = get_space_cardinality(self.obs_space)
        self.n_actions = get_space_cardinality(self.action_space)
        self.policy = np.ones((self.n_states, self.n_actions), dtype=float)
        self.policy /= self.n_actions
        dim = self.n_states*self.n_actions

        (self.W, self.b) = rbf_kernel(dim, dim, self.rng)

    def policy_sample(self, s):
        """ Samples random action from current policy """
        s_ = vec_to_int(s, self.obs_space)
        assert 0 <= s_ < self.n_states, f"invalid s={s} ({self.n_states} states)"
        eps = self.params.get("eps_explore", 0)
        assert 0 <= eps <= 1

        pi = (1.-eps)*self.policy[s_] + eps/self.n_actions
        return self.rng.choice(self.n_actions, p=pi)

    def policy_evaluate(self):
        """ Estimates Q function and stores in self.Q_est """
        if self.params.get("train_method", "mc") == "mc":
            self.monte_carlo_Q()
        elif self.params["train_method"] == "ctd":
            self.ctd_Q()
        elif self.params["train_method"] == "vrftd":
            self.vrftd_Q()

    def monte_carlo_Q(self):
        """ 
        Constructs Q/advantage function by Monte-Carlo. If missing any data
        points, we use RKHS to fill missing points.
        """
        # check if we are missing any points 
        Q_est, Ind_est = self.rollout.compute_all_stateaction_value()

        assert Q_est.shape == Ind_est.shape
        assert len(Q_est.shape) == 2

        if self.params.get("verbose", 0) >= 1:
            visited = np.sum(Ind_est)
            total = np.prod(np.shape(Ind_est))
            print(f"Visited {visited}/{total} sa pairs")

        self.Q_est = Q_est
        self.Ind_est = Ind_est

    def ctd_Q(self):
        raise NotImplemented

    def vrftd_Q(self):
        raise NotImplemented

    def policy_update(self): 
        if self.params.get("entropy", "kl").lower() == "kl":
            self.kl_policy_update()
        elif self.params["entropy"].lower() == "tsallis":
            self.tsallis_policy_update()
        else:
            self.kl_policy_update()

    def kl_policy_update(self):
        """ Policy update with PMD and KL divergence """
        eta = self.get_stepsize_schedule()
        G = np.copy(self.Q_est)
        policy_threshold = self.n_actions**(-2/(1-self.params["gamma"]))
        self.policy_update_gradient_processing(G, policy_threshold)
        G -= np.atleast_2d(np.max(G, axis=1)).T
        self.policy = np.multiply(self.policy, np.exp(eta*G))

        safe_normalize_row(self.policy)

    def policy_update_gradient_processing(self, G, policy_threshold):
        """ 
        Sets Q-function large (in-place) for rarefly visited state-action pairs
        """
        below_threshold_idxs = np.where(self.policy <= policy_threshold)
        if np.sum(below_threshold_idxs) == 0:
            return 
        upper_vals = np.maximum(np.max(G, axis=1), 1./(1-self.params["gamma"]))
        upper_vals = np.repeat(np.expand_dims(upper_vals, axis=1), self.n_actions, axis=1)
        np.put(G, below_threshold_idxs, uppder_vals)

    def tsallis_policy_update(self):
        """ Policy update with PMD and Tsallis divergence (with p=1/2) """
        policy_threshold = (1-self.params["gamma"])**2 * self.n_actions**(-1)
        warnings.warn("Tsallis entropy not yet implemented, not policy update")
        pass

    def policy_performance(self) -> float: 
        """ Estimate value function """
        if not hasattr(self, "Q_est"):
            return np.inf
        return np.mean(self.get_value_function())

    def get_value_function(self):
        return np.sum(np.multiply(self.Q_est, self.policy), axis=1)

class PMDGeneralStateFiniteAction(FOPO):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.check_PMDGeneralStateFiniteAction_params()
        self.action_space = env.action_space
        self.obs_space = env.observation_space
        (action_is_finite, action_dim, _) = get_space_property(self.action_space)
        (obs_is_finite, obs_dim, _) = get_space_property(self.obs_space)

        assert action_is_finite, "Action space not finite"

        self.n_actions = get_space_cardinality(self.action_space)
        if "dim" not in params:
            warnings.warn("Did not pass 'dim' into params, setting dim=100")
        # TODO: Use this for function approximation
        dim = params.get("dim", 100)
        if isinstance(obs_dim, tuple):
            assert len(obs_dim) == 1, "Can only handle 1D observation space"
            self.obs_dim = obs_dim[0]
        if isinstance(action_dim, tuple):
            assert len(action_dim) == 1, "Can only handle 1D action space"
            action_dim = action_dim[0]

        assert action_dim == 1, f"action dim={action_dim}"

        # Env normalization
        self.obs_runstat = RunningStat(obs_dim)
        self.action_runstat = RunningStat(self.n_actions)
        self.rwd_runstat = RunningStat(1)
        self.updated_at_least_once = False

        # uniform policy and function approximation
        self.env_warmup()
        (_, _, s_visited, _) = self.rollout.get_est_stateaction_value()
        X = s_visited
        self.fa = LinearFunctionApproximator(self.n_actions, X, params)
        self.last_thetas = np.zeros((self.n_actions, self.fa.dim), dtype=float)
        self.last_intercepts = np.zeros((self.n_actions, 1), dtype=float)
        self.theta_accum = np.copy(self.last_thetas)
        self.intercept_accum = np.copy(self.last_intercepts)
        self.last_theta_accum = np.copy(self.theta_accum)
        self.last_intercept_accum = np.copy(self.intercept_accum)

        self.last_max_q_est = ...
        self.last_max_adv_est = ...

    def check_PMDGeneralStateFiniteAction_params(self):
        # TODO: Remove this is doing NN or SGD
        return

    def _get_policy(self, s):
        log_policy_at_s = np.zeros(self.n_actions, dtype=float)
        for i in range(self.n_actions):
            self.fa.set_coef(self.theta_accum[i], i)
            self.fa.set_intercept(self.intercept_accum[i], i)
            log_policy_at_s[i] = self.fa.predict(np.atleast_2d(s), i)
        # TEMP (more robust way to do regularization)
        mu_h = self.params.get("mu_h", 0)
        log_policy_at_s *= (mu_h+1)**(-self.t)
        policy_at_s = np.exp((log_policy_at_s - np.max(log_policy_at_s)))
        policy_at_s = np.atleast_2d(policy_at_s)

        safe_normalize_row(policy_at_s)

        return np.squeeze(policy_at_s)

    def policy_sample(self, s):
        """ Samples random action from current policy """
        if not self.updated_at_least_once:
            return self.rng.integers(self.n_actions)

        eps = self.params.get("eps_explore", 0)
        assert 0 <= eps <= 1

        pi = (1.-eps)*self._get_policy(s) + eps/self.n_actions
        return self.rng.choice(self.n_actions, p=pi)

    def policy_evaluate(self):
        """ Estimates Q function and stores in self.Q_est """
        self.obs_runstat.update()
        self.action_runstat.update()
        self.rwd_runstat.update()

        if self.params.get("train_method", "mc") == "mc":
            self.monte_carlo_Q()
        elif self.params["train_method"] == "ctd":
            self.ctd_Q()
        elif self.params["train_method"] == "vrftd":
            self.vrftd_Q()

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

        self.last_max_q_est = np.max(np.abs(q_est))
        self.last_max_adv_est = np.max(np.abs(adv_est))

        X = s_visited
        y = adv_est if self.params["use_advantage"] else q_est

        # TODO: Make this a setting
        if self.params["normalize_sa_val"]:
            y = (y - np.mean(y))/(np.std(y) + 1e-8)

        # extract fitted parameters
        not_visited_actions = []
        loss = 0.
        for i in range(self.n_actions):
            action_i_idx = np.where(a_visited==i)[0]
            if len(action_i_idx) == 0:
                not_visited_actions.append(i)
                continue
            loss += self.fa.update(X[action_i_idx], y[action_i_idx], i)
            self.last_thetas[i] = np.copy(self.fa.get_coef(i))
            self.last_intercepts[i] = np.copy(self.fa.get_intercept(i))
        if len(not_visited_actions) > 0:
            print(f"Did not update actions {not_visited_actions}")

        self.last_pe_loss = loss/self.n_actions

    def ctd_Q(self):
        raise NotImplemented

    def vrftd_Q(self):
        raise NotImplemented

    def policy_update(self): 
        if self.params.get("entropy", "kl").lower() == "kl":
            self.kl_policy_update()
        elif self.params["entropy"].lower() == "tsallis":
            self.tsallis_policy_update()
        else:
            self.kl_policy_update()

    def kl_policy_update(self):
        """ Policy update with PMD and KL divergence """
        self.updated_at_least_once = True
        # TEMP
        # self.theta_accum += self.get_stepsize_schedule()*self.last_thetas
        # self.intercept_accum += self.get_stepsize_schedule()*self.last_intercepts
        eta = self.get_stepsize_schedule()

        max_grad_norm = float(self.params["max_grad_norm"])
        inf_norms_theta = np.max(np.abs(self.last_thetas), axis=1)
        inf_norms_int = np.abs(self.last_intercepts)
        inf_norms = np.maximum(inf_norms_theta, inf_norms_int)
        if max_grad_norm > 0:
            clip_coef = np.clip(np.divide(max_grad_norm, inf_norms), a_min=0, a_max=1.)
        else:
            clip_coef = np.ones(len(inf_norms))

        self.last_theta_accum = np.copy(self.theta_accum)
        self.last_intercept_accum = np.copy(self.intercept_accum)
        self.theta_accum += eta * np.diag(clip_coef)@self.last_thetas
        self.intercept_accum += eta * np.diag(clip_coef)@self.last_intercepts

    def tsallis_policy_update(self):
        """ Policy update with PMD and Tsallis divergence (with p=1/2) """
        raise NotImplemented

    def policy_performance(self) -> float: 
        """ Estimate value function """
        if not hasattr(self, "Q_est"):
            return 0
        return np.mean(self.get_value_function())

    def get_value_function(self):
        return np.sum(np.multiply(self.Q_est, self.policy), axis=1)

    def env_warmup(self):
        """ 
        Runs the environment for a fixed number of iterations estimate
        empirical mean and variance of observations, actions, and rewards
        """
        old_rollout_len = self.params["rollout_len"]
        self.params["rollout_len"] =  10
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
        q_s = []
        for i in range(self.n_actions):
            self.fa.set_coef(self.last_thetas[i], i)
            self.fa.set_intercept(self.last_intercepts[i], i)
            q_s.append(self.fa.predict(np.atleast_2d(s), i))
        return np.dot(q_s, self._get_policy(s))

    def prepare_log(self):
        l = 15
        super().prepare_log()

        coef_change = la.norm(self.last_theta_accum - self.theta_accum)
        bias_change = la.norm(self.last_intercept_accum - self.intercept_accum)

        self.msg += "train/\n"
        self.msg += f"  {'pe_loss':<{l}}: {self.last_pe_loss:.4e}\n"
        self.msg += f"  {'stepsize':<{l}}: {self.get_stepsize_schedule():.4e}\n"
        self.msg += f"  {'delta_coef':<{l}}: {coef_change:.4e}\n"
        self.msg += f"  {'delta_bias':<{l}}: {bias_change:.4e}\n"

# class PMDGeneralStateAction(PMD):

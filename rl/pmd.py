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
from rl import NNFunctionApproximator
from rl.utils import vec_to_int
from rl.utils import safe_normalize_row
from rl.utils import rbf_kernel
from rl.utils import get_rbf_features
from rl.utils import get_space_property
from rl.utils import get_space_cardinality
from rl.utils import pretty_print_gridworld
from rl.utils import RunningStat
from rl.utils import tsallis_policy_update

import torch as th

from stable_baselines3.common.utils import obs_as_tensor

class FOPO(RLAlg):
    def __init__(self, env, params):
        super().__init__(env, params)

        # initialize
        self.t = -1
        self._last_obs, _ = env.reset()

        self.rollout = Rollout(env, **params)
        self.episode_rewards = []

        self.n_episodes = 0
        self.n_step = 0
        self.last_iter_ep = 0
        self.max_episodes = params['max_episodes'] if params['max_episodes'] > 0 else np.inf
        self.max_steps = params['max_steps'] if params['max_steps'] > 0 else np.inf
        self.emp_Q_max_arr = []
        self.sto_Q_max_arr = []
        self.curr_beta_sum = 0
        self.prev_beta_sum = 0
        self.prev_lam_t = 0

    def _learn(self, max_iter):
        """ Runs PMD algorithm for `max_iter`s """
        checkpoint = np.linspace(0, max_iter, num=min(10, max_iter), dtype=int)
        self.s_time = time.time()

        for t in range(max_iter):
            self.t = t

            (beta_t, lam_t) = self.get_stepsize_schedule()
            self.curr_beta_sum += beta_t

            if self.params["pmd_sb3_policy"]:
                self.sb3_collect_rollouts()
            else:
                self.collect_rollouts()

            if self.n_episodes == self.max_episodes:
                break

            self.policy_evaluate()

            self.policy_update()

            self.prepare_log()
            self.dump_log()
            self.prev_beta_sum = self.curr_beta_sum
            self.prev_lam_t = lam_t

            if t in checkpoint:
                self.save_episode_reward_and_len()
                # self.save_policy_change()

            if self.n_episodes >= self.max_episodes or self.n_step >= self.max_steps:
                break

        self.save_episode_reward_and_len()
        # self.save_policy_change()
        # return moving average reward (length 100)
        ep_cum_rwds = self.rollout.get_ep_rwds()
        t = min(100, len(ep_cum_rwds))
        return np.mean(ep_cum_rwds[-t:])

    def collect_rollouts(self):
        """ Collect samples for policy evaluation """
        self.rollout.clear_batch()

        s = self._last_obs

        for t in range(self.params["pmd_rollout_len"]): 
            v_s = self.estimate_value(s)

            a = self.policy_sample(s)
            (next_s, r_raw, term, trunc, _)  = self.env.step(a)
            done = term or trunc
            r = self.normalize_rwd(r_raw)
            # if truncated due to time limit, bootstrap cost-to-go
            if trunc:
                v_next_s = self.estimate_value(next_s)
                r += self.params["gamma"] * v_next_s
            if t == self.params["pmd_rollout_len"]-1:
                self._curr_s = next_s
                self._last_pred_value = self.estimate_value(next_s)
                self._last_state_done = done
            self.rollout.add_step_data(s, a, r, done, v_s, r_raw=r_raw)

            self.n_step += 1

            if done:
                self.n_episodes += 1
                if self.n_episodes == self.max_episodes or self.n_step == self.max_steps:
                    return 
                s, _ = self.env.reset()
                next_s = s

            self._last_obs = np.copy(s)
            s = next_s

    def sb3_collect_rollouts(self):
        self.collect_rollouts()

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
        """ Returns stepsize for both PDA and PMD.

        The step sizes $(beta_t, lam_t)$ are from PDA; We can set $eta_t = beta_t / lam_t$ in PMD.
        """
        stepsize_base = self.params["pmd_stepsize_base"]
        if self.params["pmd_stepsize_type"] == "pmd":
            beta_t = 1./np.sqrt(self.params["max_iters"])
            lam_t = 1./stepsize_base
        # elif self.params["stepsize"]  == "adapt_constant":
        #     # TODO: Incorporate regularization
        #     Q_inf = abs(self.emp_Q_max_arr[-1])
        #     tQ_inf_sq = self.stoch_Q_second_moment()
        #     M_h = self.params.get("M_h",0)
        #     lam_t = 1./np.sqrt(2*np.log(self.n_actions)/(Q_inf**2 + tQ_inf_sq))
        #     beta_t = 1./np.sqrt(self.params["max_iter"])
        elif self.params["pmd_stepsize_type"] == "pda_1":
            beta_t = self.t+1
            lam_t = (self.t+1)**(1.5)/stepsize_base
        elif self.params["pmd_stepsize_type"] == "pda_2":
            beta_t = self.t+1
            k = self.params["max_iters"]
            lam_t = k *(k+1)/stepsize_base
        # elif self.params["stepsize"] == "adapt_decreasing":
        #     Q_inf = abs(self.emp_Q_max_arr[-1])
        #     tQ_inf_sq = self.stoch_Q_second_moment()
        #     M_h = self.params.get("M_h",0)
        #     lam_t = np.sqrt(2*np.log(self.n_actions)/(Q_inf**2 + tQ_inf_sq))
        #     beta_t = t+1
        else:
            raise Exception("Unknown stepsize rule %s" % self.params['pmd_stepsize_type'])

        return (beta_t, lam_t)

    def stoch_Q_second_moment(self, mv_avg_len=5):
        l = min(mv_avg_len, len(self.sto_Q_max_arr))
        if l == 0: return 0
        return np.mean(np.square(self.sto_Q_max_arr[-l:]))

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

        if self.params['pmd_sb3_policy']:
            rwd_arr_trunc = [ep_info["r"] for ep_info in self.sb3_policy.ep_info_buffer]
            len_arr_trunc = [ep_info["l"] for ep_info in self.sb3_policy.ep_info_buffer]
            moving_avg = np.mean(rwd_arr_trunc)
        else:
            rwd_arr = self.rollout.get_ep_rwds()
            len_arr = self.rollout.get_ep_lens()
            rwd_arr_trunc = rwd_arr[self.last_iter_ep: self.n_episodes]
            len_arr_trunc = len_arr[self.last_iter_ep: self.n_episodes]
            i = min(25, len(rwd_arr))
            moving_avg = np.mean(rwd_arr[-i:])

        self.msg += "rollout/\n"
        self.msg += f"  {'itr_ep_len_mean':<{l}}: {np.mean(len_arr_trunc):.2f}\n"
        self.msg += f"  {'itr_ep_rwd_mean':<{l}}: {np.mean(rwd_arr_trunc):.2f}\n"
        self.msg += f"  {'itr_n_ep':<{l}}: {int(self.n_episodes-self.last_iter_ep)}\n"
        self.msg += f"  {'ep_rwd_25-ma':<{l}}: {moving_avg:.2f}\n"

        self.msg += "time/\n"
        self.msg += f"  {'itr':<{l}}: {self.t}\n"
        self.msg += f"  {'time_elap':<{l}}: {time.time()-self.s_time:.2f}\n"
        self.msg += f"  {'tot_steps':<{l}}: {self.n_step}\n"
        self.msg += f"  {'tot_ep':<{l}}: {self.n_episodes}\n"

        self.last_iter_ep = self.n_episodes

    def dump_log(self):
        self.msg += "-"*30 + "\n"
        print(self.msg, end="")
        self.msg = ""

    def save_episode_reward_and_len(self):
        rwd_arr = self.rollout.get_ep_rwds()
        len_arr = self.rollout.get_ep_lens()

        fmt="%1.2f,%i"
        arr = np.vstack((np.atleast_2d(rwd_arr), np.atleast_2d(len_arr))).T
        with open(self.params["log_file"], "wb") as fp:
            fp.write(b"episode rewards,episode len\n")
            np.savetxt(fp, arr, fmt=fmt)
        print("Saved episode data to %s" % self.params['log_file'])

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

        pi = self.policy[s_] 
        return self.rng.choice(self.n_actions, p=pi)

    def policy_evaluate(self):
        """ Estimates Q function and stores in self.Q_est """
        self.monte_carlo_Q()

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
        self.exp_policy_update()

    def exp_policy_update(self):
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
        self.action_space = env.action_space
        self.obs_space = env.observation_space
        (action_is_finite, action_dim, _) = get_space_property(self.action_space)
        (obs_is_finite, obs_dim, _) = get_space_property(self.obs_space)

        assert action_is_finite, "Action space not finite"

        self.n_actions = get_space_cardinality(self.action_space)
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
        if self.params["pmd_fa_type"] == "linear":
            print("Using linear function approximation with RKHS")
            self.fa_Q = LinearFunctionApproximator(X, params)
            self.last_thetas = np.zeros((self.n_actions, self.fa_Q.dim), dtype=float)
            self.last_intercepts = np.zeros((self.n_actions, 1), dtype=float)
            self.theta_accum = np.copy(self.last_thetas)
            self.intercept_accum = np.copy(self.last_intercepts)
            self.last_theta_accum = np.copy(self.theta_accum)
            self.last_intercept_accum = np.copy(self.intercept_accum)
        else:
            print("Using neural network with tanh activation")
            self.fa_Q = NNFunctionApproximator(
                input_dim=self.obs_dim, 
                output_dim=self.n_actions, 
                params=params
            )
            self.fa_Q_accum = NNFunctionApproximator(
                input_dim=self.obs_dim, 
                output_dim=self.n_actions, 
                params=params
            )

        self.sb3_policy = None
        if self.params["pmd_sb3_policy"]:
            from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
            from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
            from stable_baselines3.common.buffers import RolloutBuffer
            self.sb3_policy = OnPolicyAlgorithm(
                policy=ActorCriticPolicy,
                env=self.env,
                learning_rate=params['ppo_lr'],
                n_steps=params['ppo_rollout_len'],
                gamma=params['gamma'],
                gae_lambda=params['ppo_gae_lambda'],
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=params['ppo_max_grad_norm'],
                use_sde=False,
                sde_sample_freq=-1,
                # rollout_buffer_class=None,
                # rollout_buffer_kwargs=None,
                stats_window_size=100,
                tensorboard_log=None,
                policy_kwargs=None,
                verbose=0,
                device='cpu',
                seed=params['seed'],
                _init_setup_model=True,
                supported_action_spaces=(
                    gym.spaces.Box,
                    gym.spaces.Discrete,
                ),
            )

            # TODO: What should I do with these two outputs?
            total_timesteps, callback = self.sb3_policy._setup_learn(
                total_timesteps=params['max_steps'],
                callback=None,
                reset_num_timesteps=True,
                tb_log_name="OnPolicyAlgorithm",
                progress_bar=False,
            )

            self.rollout_buffer = RolloutBuffer(
                params['ppo_rollout_len'],
                self.obs_space,
                self.action_space,
                device='cpu',
                gamma=params['gamma'],
                gae_lambda=params['ppo_gae_lambda'],
                n_envs=1,
            )

        self.last_max_q_est = ...
        self.last_max_adv_est = ...
        self.last_policy_at_s = np.ones(self.n_actions, dtype=float)/self.n_actions
        self.last_dual_tsallis = None

        self._last_s_visited = ...
        self._last_a_visited = ...
        self._last_y = ...

    def sb3_collect_rollouts(self):
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.sb3_policy.policy.set_training_mode(False)

        n_steps = 0
        self.rollout_buffer.reset()

        # Sample new weights for the state dependent exploration
        # if self.use_sde:
        if False:
            num_envs = 1
            self.sb3_policy.policy.reset_noise(num_envs)

        if isinstance(self._last_state_done, bool):
            self._last_episode_starts = [self._last_state_done]
        if len(self._last_obs.shape) < 2:
            self._last_obs = self._last_obs.reshape(1,-1)

        while n_steps < self.params["pmd_rollout_len"]:
            # if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            if False:
                # Sample a new noise matrix
                num_envs = 1
                self.sb3_policy.policy.reset_noise(num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.sb3_policy.device)
                actions, values, log_probs = self.sb3_policy.policy(obs_tensor.unsqueeze(0)) 

            actions = np.array([self.policy_sample(self._last_obs.squeeze())])

            new_obs, rewards, dones, infos = self.sb3_policy.env.step(actions)

            self.n_step += 1
            n_steps += 1

            self.sb3_policy._update_info_buffer(infos)

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.sb3_policy.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.sb3_policy.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.params['gamma'] * terminal_value
                if done:
                    self.n_episodes += 1

            self.rollout_buffer.add( 
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.sb3_policy.policy.predict_values(obs_as_tensor(new_obs, self.sb3_policy.device))  # type: ignore[arg-type]

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        """
        # TEMP: For human baseline to see change in policy and policy evaluation
        try:
            # this is for discrete
            N = int(self.env.observation_space.high_repr)
            n = self.env.observation_space._shape[0]
            n_s = N*n
        except:
            # this is for box
            n_s = self.env.observation_space._shape[0]

        n_a = self.env.action_space.n
        # coords = np.meshgrid(*((np.arange(N),)*n))
        # self.monitor_s_enum = np.vstack([coords[i].flatten() for i in range(len(coords))]).T
        n_samples = 50
        self.monitor_s_enum = np.array([self.env.observation_space.sample() for _ in range(n_samples)])

        # [iteration; state; q(s,a); pi(s,a)]
        self._policy_prog = np.zeros((1024, 1+n_s+3*n_a), dtype=float)
        self._pp_ct = 0
        self.record_policy_change()
        """

    def _get_logpolicy(self, s):
        if not self.updated_at_least_once:
            return np.zeros(self.n_actions, dtype=float)

        log_policy_at_s = np.zeros(self.n_actions, dtype=float)
        mu_h = self.params["pmd_mu_h"]
        (beta_t, lam_t) = self.get_stepsize_schedule()
        if self.params["pmd_fa_type"] == "linear":
            for i in range(self.n_actions):
                self.fa_Q.set_coef(self.theta_accum[i])
                self.fa_Q.set_intercept(self.intercept_accum[i])
                log_policy_at_s[i] = self.fa_Q.predict(np.atleast_2d(s))
        elif self.params["pmd_stepsize_type"] == "pmd":
            # Since fa_Q_accum learns:
            # $(\eta_sum_k)^{-1}\sum_{t=0}^k (1+eta mu)^{k-t} \eta Q(s,a;theta_t)
            # where $\eta_sum_k = (k+1) * eta/(1+eta*mu)$
            eta = beta_t/lam_t
            eta_sum = (self.t+1)*eta/(1+eta*mu_h)
            log_policy_at_s = eta_sum * self.fa_Q_accum.predict(
                np.atleast_2d(s),
                np.arange(self.n_actions),
            )
        elif "pda" in self.params["pmd_stepsize_type"]:
            # Since fa_Q_accum learns:
            # $(beta_sum)^{-1}\sum_{t=0}^k \beta_t Q(s,a;\theta_t)$,
            # we need to scale it back
            alpha = self.curr_beta_sum/(self.curr_beta_sum*mu_h + lam_t)
            log_policy_at_s = alpha * self.fa_Q_accum.predict(
                np.atleast_2d(s),
                np.arange(self.n_actions),
            )
        else:
            raise Exception("Unknown pmd step size type %s" % self.params["pmd_stepsize_type"])

        return log_policy_at_s

    def _get_policy(self, s):
        if not self.updated_at_least_once:
            return np.ones(self.n_actions, dtype=float)/self.n_actions

        log_policy_at_s = self._get_logpolicy(s)

        if self.params['pmd_policy_divergence'] == 'kl':
            policy_at_s = np.exp((log_policy_at_s - np.max(log_policy_at_s)))
            policy_at_s = np.atleast_2d(policy_at_s)
        elif self.params['pmd_policy_divergence'] == 'tsallis':
            # we can only use Tsallis divergence with PDA
            assert "pda" in self.params["pmd_stepsize_type"], "Can only use Tsallis divergence with PDA"

            mu_h = self.params["pmd_mu_h"]
            (beta_t, lam_t) = self.get_stepsize_schedule()
            # need to remove scaling factor (see `_get_logpolicy`)
            alpha = self.curr_beta_sum*mu_h + lam_t
            grad_sum = alpha * log_policy_at_s 
            policy_at_s, dual_tsallis = tsallis_policy_update(
                -grad_sum, # take negative since we want to maximize
                1./lam_t,  # 1/lam_t since tsallis update takes reciprocal of it
                self.last_dual_tsallis,
            )
            policy_at_s = np.expand_dims(policy_at_s, axis=0)
            self.last_dual_tsallis = dual_tsallis
        else:
            raise Exception("Unknown policy divergence %s" % self.params['pmd_policy_divergence'])

        safe_normalize_row(policy_at_s)

        return np.squeeze(policy_at_s)

    def policy_sample(self, s):
        """ Samples random action from current policy """
        pi = self._get_policy(self.normalize_obs(s))
        return self.rng.choice(self.n_actions, p=pi)

    def policy_evaluate(self):
        """ Estimates Q function and stores in self.Q_est """
        # self.obs_runstat.update()
        # self.action_runstat.update()
        # self.rwd_runstat.update()

        self.monte_carlo_Q()
        # TODO: Import rollout

    def monte_carlo_Q(self):
        """ 
        Constructs Q/advantage function by Monte-Carlo. If missing any data
        points, we use RKHS to fill missing points.

        We update our running stats (e.g., mean and variance) for next
        iteration
        """
        # check if we are missing any points 
        if self.params['pmd_sb3_policy']:
            q_est = adv_est = self.rollout_buffer.advantages.squeeze()
            s_visited = self.rollout_buffer.observations.squeeze()
            a_visited = self.rollout_buffer.actions.squeeze()
        else:
            (q_est, adv_est, s_visited, a_visited) = self.rollout.get_est_stateaction_value(
                self._last_pred_value,
                self._last_state_done,
            )

        X = np.array([self.normalize_obs(s) for s in s_visited])
        y = adv_est if self.params["pmd_use_adv"] else q_est
        if self.params['pmd_normalize_sa_val']:
            y = (y-np.mean(y))/(np.std(y) + 1e-8)
        self.last_max_q_est = np.max(np.abs(q_est))
        self.last_max_adv_est = np.max(np.abs(adv_est))
        self.emp_Q_max_arr.append(np.max(y))

        self._last_s_visited = np.copy(X)
        self._last_a_visited = np.copy(a_visited)
        self._last_y = np.copy(y)

        # extract fitted parameters
        not_visited_actions = []
        loss = 0.
        sto_Q_arr = [0]
        if self.params['pmd_fa_type'] == "linear":
            for i in range(self.n_actions):
                action_i_idx = np.where(a_visited==i)[0]
                if len(action_i_idx) == 0:
                    not_visited_actions.append(i)
                    continue
                self.fa_Q.set_coef(self.last_thetas[i])
                self.fa_Q.set_intercept(self.last_intercepts[i])
                train_loss, _ = self.fa_Q.update(X[action_i_idx], y[action_i_idx])
                pred = self.fa_Q.predict(X[action_i_idx])
                sto_Q_arr.append(np.mean(np.square(pred)))
                self.last_thetas[i] = self.fa_Q.get_coef()
                self.last_intercepts[i] = self.fa_Q.get_intercept()
        else:
            train_loss,_ = self.fa_Q.update(X, a_visited, y, validation_frac=0)

        if isinstance(train_loss, float) or isinstance(train_loss, int):
            loss += train_loss 
        else:
            # TODO: What's going on here
            loss += np.mean(train_loss)
        if len(not_visited_actions) > 0:
            print(f"Did not update actions {not_visited_actions}")

        self.sto_Q_max_arr.append(np.max(sto_Q_arr))
        if isinstance(loss, np.ndarray):
            loss = np.sum(loss)
        self.last_pe_loss = loss/self.n_actions

    def policy_update(self): 
        self.exp_policy_update()

        # TEMP
        # self.record_policy_change()

    def exp_policy_update(self):
        self.updated_at_least_once = True
        if self.params['pmd_sb3_policy']:
            self.sb3_policy_network_update()
        if self.params['pmd_fa_type'] == "linear":
            self.exp_policy_update_linear()
        elif self.params['pmd_fa_type'] == "nn":
            self.exp_policy_update_nn()
        else:
            raise Exception(f"Unknown function approximation {self.params['fa_type']}")

    def exp_policy_update_linear(self):
        """
        For entropy regularization with strong convexity $mu_h$, PDA has the explicit argmin solution of
        $$
            \pi_{k+1}(s) 

            \propto exp{ -\sum_{t=0}^k \beta_t \phi(s,*;\theta_t)/(\bar{\beta}_k * \mu_h + \lambda_k) }
        $$
        (for argmax, remove negative sign with a positive sign) while PMD has
        $$
            \pi_{k+1}(s) 
            \propto exp{ (1+\eta_t*\mu_h)^{-1} * (log(\pi_k(s)) - \eta_t \phi(s,*;\theta_t)) }.
            \propto exp{ -\sum_{t=0}^k (\pi_{\tau=t}^k (1+\eta_t*\mu_h))^{-1} * \eta_t \phi(s,*\theta_t) }
        $$
        """
        (beta_t, lam_t) = self.get_stepsize_schedule()
        mu_h = self.params['pmd_mu_h']
        self.last_theta_accum = np.copy(self.theta_accum)
        self.last_intercept_accum = np.copy(self.intercept_accum)

        if self.params['pmd_stepsize_type'] == "pmd":
            eta_t = beta_t/lam_t # see self.get_stepsize_schedule
            alpha_t = 1+eta_t*mu_h

            self.theta_accum = 1./alpha_t * (self.theta_accum + eta_t * self.last_thetas)
            self.intercept_accum = 1./alpha_t * (self.intercept_accum + eta_t * self.last_intercepts)
        else:
            curr_alpha_t = self.curr_beta_sum*mu_h + lam_t
            prev_alpha_t = self.prev_beta_sum*mu_h + self.prev_lam_t

            self.theta_accum = (prev_alpha_t/curr_alpha_t) * self.theta_accum \
                               + (beta_t/curr_alpha_t) * self.last_thetas
            self.intercept_accum = (prev_alpha_t/curr_alpha_t) * self.intercept_accum  \
                                    + (beta_t/curr_alpha_t) * self.last_intercepts

    def sb3_policy_network_update(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.sb3_policy.policy.set_training_mode(True)
        # Update optimizer learning rate
        self.sb3_policy._update_learning_rate(self.sb3_policy.policy.optimizer)

        value_losses = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.params['pmd_pe_max_epochs']):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.params['pmd_batch_size']):
                actions = rollout_data.actions
                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.sb3_policy.use_sde:
                    self.sb3_policy.policy.reset_noise(self.batch_size)

                values, _, _ = self.sb3_policy.policy.evaluate_actions(rollout_data.observations, actions) 
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                # if self.params['ppo_normalize_adv'] and len(advantages) > 1:
                #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Value loss using the TD(gae_lambda) target
                values_pred = values
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                loss = self.sb3_policy.vf_coef * value_loss

                # Optimization step
                self.sb3_policy.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.sb3_policy.policy.parameters(), self.params['pmd_max_grad_norm'])
                self.sb3_policy.policy.optimizer.step()

            if not continue_training:
                break

    def exp_policy_update_nn(self):
        """ Policy update with PDA and KL divergence """
        (beta_t, lam_t) = self.get_stepsize_schedule()
        mu_h = self.params["pmd_mu_h"]
        # corresponds to PDA
        # curr_alpha_t = self.curr_beta_sum*mu_h + lam_t
        # prev_alpha_t = self.prev_beta_sum*mu_h + self.prev_lam_t

        if self.params['pmd_stepsize_type'] == 'pmd':
            eta_t = beta_t/lam_t
            alpha = 1+eta_t*mu_h
            alpha_1 = 1./alpha * self.t/(self.t+1)
            alpha_2 = 1./alpha * 1./(self.t+1)
        else:
            alpha_1 = self.prev_beta_sum/self.curr_beta_sum
            alpha_2 = beta_t/self.curr_beta_sum

        y        = self._last_y 
        # y        = self.fa_Q.predict(X_i)[:,i]
        Q_acc_pred = self.fa_Q_accum.predict(self._last_s_visited, self._last_a_visited)
        target = alpha_1 * Q_acc_pred + alpha_2*y

        train_losses, test_losses = self.fa_Q_accum.update(
            self._last_s_visited, 
            self._last_a_visited,
            target,
            validation_frac=0,
        )

        self.last_po_loss = np.mean(train_losses)
        self.last_po_test_loss = np.mean(test_losses)
        return train_losses, test_losses

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
        old_rollout_len = self.params["pmd_rollout_len"]
        self.params["pmd_rollout_len"] = max(2048, old_rollout_len)
        self.collect_rollouts()
        self.params["pmd_rollout_len"] =  old_rollout_len

        (q_est, adv_est, s_visited, a_visited) = self.rollout.get_est_stateaction_value()
        [self.normalize_obs(s) for s in s_visited]
        y = adv_est if self.params["pmd_use_adv"] else q_est
        [self.normalize_rwd(y_i) for y_i in y]
        # if self.params.get("normalize_sa_val", None):
        #     y = (y-np.mean(y))/(np.std(y) + 1e-8)
        self.last_max_q_est = np.max(np.abs(q_est))
        self.last_max_adv_est = np.max(np.abs(adv_est))
        self.emp_Q_max_arr.append(np.max(y))
        self.sto_Q_max_arr.append(np.max(y))

        self.obs_runstat.update()
        self.action_runstat.update()
        self.rwd_runstat.update()

        print("Finished normalization warmup")

    def normalize_obs(self, s):
        self.obs_runstat.push(s)
        if self.params["pmd_normalize_obs"]:
            return np.divide(s-self.obs_runstat.mean, np.sqrt(self.obs_runstat.var))
        return s

    def normalize_rwd(self, r):
        """ Only scale reward, do not re-center """
        self.rwd_runstat.push(r)
        if self.params["pmd_normalize_rwd"]:
            return np.divide(r, np.sqrt(self.rwd_runstat.var) + abs(self.rwd_runstat.mean))
            # return np.divide(r-self.rwd_runstat.mean, np.sqrt(self.rwd_runstat.var**0.5))
        return r

    def get_q_s(self, s):
        if not self.updated_at_least_once:
            return np.zeros(self.n_actions)
        q_s = []
        if self.params["pmd_fa_type"] == "linear":
            for i in range(self.n_actions):
                self.fa_Q.set_coef(self.last_thetas[i])
                self.fa_Q.set_intercept(self.last_intercepts[i])
                q = self.fa_Q.predict(np.atleast_2d(s))
                q_s.append(q)
        else:
            q_s = self.fa_Q.predict(np.atleast_2d(s), np.arange(self.n_actions))
        return np.squeeze(np.array(q_s))

    def estimate_value(self, s):
        if not self.updated_at_least_once:
            return 0
        q_s = self.get_q_s(s)
        return np.dot(q_s, self._get_policy(s))

    def prepare_log(self):
        l = 15
        super().prepare_log()

        policy_at_s = self._get_policy(self._last_obs)

        self.msg += "train/\n"
        self.msg += f"  {'pe_loss':<{l}}: {self.last_pe_loss:.4e}\n"
        if self.params["pmd_fa_type"] == "nn":
            self.msg += f"  {'po_train_loss':<{l}}: {self.last_po_loss:.4e}\n"
            self.msg += f"  {'po_test_loss':<{l}}: {self.last_po_test_loss:.4e}\n"
        if self.params.get("stepsize", "constant") == "constant":
            (beta_t, lam_t) = self.get_stepsize_schedule()
            eta_t = beta_t/lam_t
            self.msg += f"  {'stepsize':<{l}}: {eta_t:.4e}\n"
        else:
            (beta_t, lam_t) = self.get_stepsize_schedule()
            self.msg += f"  {'stepsize':<{l}}: {beta_t:.2e}, {lam_t:.2e}\n"
        if self.params["pmd_fa_type"] == "linear":
            coef_change = la.norm(self.last_theta_accum - self.theta_accum)
            bias_change = la.norm(self.last_intercept_accum - self.intercept_accum)
            self.msg += f"  {'delta_coef':<{l}}: {coef_change:.4e}\n"
            self.msg += f"  {'delta_bias':<{l}}: {bias_change:.4e}\n"
        self.msg += f"  {'delta_policy':<{l}}: {self.last_policy_at_s}->{policy_at_s}\n"

        self.last_policy_at_s = self._get_policy(self._curr_s)
        self._last_obs = np.copy(self._curr_s)

    def record_policy_change(self):
        """ For human baselines to see how policy changes during updates """
        # zero based indexing
        n = self.env.observation_space._shape[0]
        n_a = self.env.action_space.n

        i_0 = self.t*len(self.monitor_s_enum)
        for i, s in enumerate(self.monitor_s_enum):
            logpi_s = self._get_logpolicy(s)
            pi_s = self._get_policy(s)
            q_s = self.get_q_s(s)
            self._policy_prog[self._pp_ct, 0] = self.t
            self._policy_prog[self._pp_ct, 1:1+len(s)] = np.copy(s)
            self._policy_prog[self._pp_ct, 1+len(s):1+len(s)+n_a] = q_s
            self._policy_prog[self._pp_ct, 1+len(s)+n_a:1+len(s)+2*n_a] = logpi_s
            self._policy_prog[self._pp_ct, 1+len(s)+2*n_a:1+len(s)+3*n_a] = pi_s
            self._pp_ct += 1

            if self._pp_ct == len(self._policy_prog):
                self._policy_prog = np.vstack((self._policy_prog, np.zeros(self._policy_prog.shape)))

    def save_policy_change(self):
        n = self.env.observation_space._shape[0]
        n_a = self.env.action_space.n

        import os
        fname = os.path.join("logs", "policy_progress.csv")
        with open(fname, "w") as fp:
            # header
            fp.write("iter;s;q_s;logpi_s;pi_s\n")

            for i in range(self._pp_ct):
                # iter
                fp.write("%i;" % self._policy_prog[i,0])
                # state
                fp.write(f"{tuple(self._policy_prog[i,1:1+n].astype('int'))};")
                # q_s
                fp.write(f"{tuple(self._policy_prog[i,1+n:1+n+n_a])};")
                # logpi_s
                fp.write(f"{tuple(self._policy_prog[i,1+n+n_a:1+n+2*n_a])};")
                # pi_s
                fp.write(f"{tuple(self._policy_prog[i,1+n+2*n_a:1+n+3*n_a])}\n")

        print("Saved policy progress data to %s" % fname)

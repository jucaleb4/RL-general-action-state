import sys

import warnings

import numpy as np

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

from rl import RLAlg

"""
From: https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
"""
class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, scaler, featurizer, env):
        self.scaler = scaler
        self.featurizer = featurizer
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset()[0])], [0])
            self.models.append(model)
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]
    
    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])

class QLearn(RLAlg):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.check_params() 

        (scaler, featurizer) = self.get_featurizer()
        self.estimator = Estimator(scaler, featurizer, env)

    def check_params(self):
        if "gamma" not in self.params:
            warnings.warn("Did not pass in 'gamma' into params, defaulting to 1.0")
            self.params["gamma"] = 1.0
        if "epsilon" not in self.params:
            warnings.warn("Did not pass in 'epsilon' into params, defaulting to 0.1")
            self.params["epsilon"] = 0.1
        if "epsilon_decay" not in self.params:
            warnings.warn("Did not pass in 'epsilon_decay' into params, defaulting to 1.0")
            self.params["epsilon_decay"] = 1.0

    def get_featurizer(self):
        observation_examples = np.array([self.env.observation_space.sample() for _ in range(10000)])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_examples)
        
        # Used to convert a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        featurizer = sklearn.pipeline.FeatureUnion([
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100))
                ])
        featurizer.fit(scaler.transform(observation_examples))
        return (scaler, featurizer)

    def make_epsilon_greedy_policy(self, epsilon, nA):
        """
        Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
        Args:
            estimator: An estimator that returns q values for a given state
            epsilon: The probability to select a random action . float between 0 and 1.
            nA: Number of actions in the environment.
    
        Returns:
            A function that takes the observation as an argument and returns
            the probabilities for each action in the form of a numpy array of length nA.
    
        """
        def policy_fn(observation):
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values = self.estimator.predict(observation)
            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A
        return policy_fn

    def _learn(self, n_iter):
        last_reward = 0
        episode_rewards = np.zeros(n_iter, dtype=float)
        episode_lens = np.zeros(n_iter, dtype=int)

        for i in range(n_iter):
            epsilon_ = self.params["epsilon"] * self.params["epsilon_decay"]**i
            n_actions = self.env.action_space.n
            policy = self.make_epsilon_greedy_policy(epsilon_, n_actions)
        
            # Print out which episode we're on, useful for debugging.
            # Also print reward for last episode
            sys.stdout.flush()
        
            # Reset the environment and pick the first action
            state = self.env.reset()[0]
        
            # Only used for SARSA, not Q-Learning
            next_action = None
        
            # One step in the environment
            t = 0
            curr_reward = 0
            while 1:
                # Choose an action to take
                # If we're using SARSA we already decided in the previous step
                if next_action is None:
                    action_probs = policy(state)
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                else:
                    action = next_action
            
                # Take a step
                next_state, reward, term, trunc, _ = self.env.step(action)
                curr_reward += reward
                done = term or trunc
    
                # TD Update
                q_values_next = self.estimator.predict(next_state)
            
                # Use this code for Q-Learning
                # Q-Value TD Target
                td_target = reward + self.params["gamma"] * np.max(q_values_next)
            
                # Use this code for SARSA TD Target for on policy-training:
                # next_action_probs = policy(next_state)
                # next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)             
                # td_target = reward + discount_factor * q_values_next[next_action]
            
                # Update the function approximator using our target
                self.estimator.update(state, action, td_target)
            
                print("\rStep {} @ Episode {}/{} (last episode:{})".format(t, i + 1, n_iter, last_reward), end="")
                
                if done:
                    last_reward = curr_reward
                    episode_rewards[i] = last_reward
                    episode_lens[i] = t
                    curr_reward = 0
                    print("")
                    break
                
                state = next_state
                t += 1

        self.save_episode_rewards(episode_rewards, episode_lens)

    def save_episode_rewards(self, rwd_arr, len_arr):
        if "fname" not in self.params:
            warnings.warn("No filename given, not saving")
            return
        fmt="%1.2f,%i"
        arr = np.vstack((np.atleast_2d(rwd_arr), np.atleast_2d(len_arr))).T
        with open(self.params["fname"], "wb") as fp:
            fp.write(b"episode rewards,episode len\n")
            np.savetxt(fp, arr, fmt=fmt)
        print(f"Saved episode data to {self.params['fname']}")

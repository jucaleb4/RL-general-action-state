import unittest

import numpy as np
import numpy.linalg as la

import gymnasium as gym
import gym_examples

from rl.utils import *

class UtilsForIntegerBox(unittest.TestCase):

    def get_GridWorld_env(self, size):
        env = gym.make(
            "gym_examples/GridWorld-v0", 
            size=size,
        )
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.TransformReward(env, lambda r : 1-r)
        return env

    def test_space_property(self):
        size = np.random.randint(5,10)
        env = self.get_GridWorld_env(size)
        output = get_space_property(env.observation_space)

        self.assertEqual(len(output), 3)
        self.assertEqual(output[0], True)
        self.assertEqual(output[1], (4,))
        self.assertEqual(output[2], int)

    def test_space_cardinality(self):
        size = np.random.randint(5,10)
        env = self.get_GridWorld_env(size)
        cardinality = get_space_cardinality(env.observation_space)

        self.assertEqual(cardinality, size**4)

    def test_vec_to_int_and_vice_versa(self):
        size = np.random.randint(5,10)
        env = self.get_GridWorld_env(size)
        obs = env.observation_space.sample()
        i   = vec_to_int(obs, env.observation_space)
        obs_from_i = int_to_vec(i, env.observation_space)

        self.assertTrue(np.allclose(obs, obs_from_i))

import unittest

import numpy as np
import numpy.linalg as la

import gymnasium as gym
import gym_examples

from rl import Rollout

class FiniteStateActionRollout(unittest.TestCase):

    def get_rollout_object(self):
        env = gym.make(
            "gym_examples/SimpleWorld-v0", 
            size=2,
        )

        gamma = 0.99
        rollout = Rollout(env, gamma)
        return env, rollout

    def test_set_gamma(self):
        """ Batch will need to resize (default size is 32) """
        (env, rollout) = self.get_rollout_object()

        gamma = 0.01
        rollout.set_gamma(gamma)
        
        self.assertEqual(rollout.gamma, gamma)

    def test_add_data(self):
        (env, rollout) = self.get_rollout_object()

        # add reset step
        s, info = env.reset()
        a = env.action_space.sample()
        rollout.add_step_data(s, a)

        (s,r,terminate, truncate, info) = env.step(a)
        a = env.action_space.sample()
        rollout.add_step_data(s, a, r, terminate, truncate)

    def test_add_many_data(self):
        """ Batch will need to resize (default size is 32) """
        (env, rollout) = self.get_rollout_object()

        # add reset step
        s, info = env.reset()
        a = env.action_space.sample()
        rollout.add_step_data(s, a)

        for _ in range(36):
            (s,r,terminate, truncate, info) = env.step(a)
            a = env.action_space.sample()
            rollout.add_step_data(s, a, r, terminate, truncate)

    def test_initialize_and_clear(self):
        (env, rollout) = self.get_rollout_object()
        self.assertEqual(rollout.iter_ct, 0)

        s, info = env.reset()
        a = env.action_space.sample()
        rollout.add_step_data(s, a)

        self.assertEqual(rollout.iter_ct, 1)

        rollout.clear_batch()

        self.assertEqual(rollout.iter_ct, 0)

    def test_compute_enumerated_stateaction_value_visit_all(self):
        """ 
        Visit all state-action pairs and check the Q function is correct.
        This is the order we plan to visit (s,a)

        (0,0)->(1,0)->(2,0)->(3,0)->(0,1)->(2,1)->(0,0)->(1,1)->(3,1)->(1,0)

        The rewards are:
             0      1      2      3      1      3      0      2      4

        Expected weighted cumulative reward should be (asterisk if first time)
          >> 15.22977263322566  (0,0)*
          >> 15.38360872042996  (1,0)*
          >> 14.528897697404    (2,0)*
          >> 12.655452219599999 (3,0)*
          >> 9.752982040000001  (0,1)*
          >> 8.841396000000001  (2,1)*
          >> 5.9004             (0,0)
          >> 5.960000000000001  (1,1)*
          >> 4.0                (3,1)*
        """
        (env, rollout) = self.get_rollout_object()

        action_plan    = [0,0,0,0,1,1,0,1,1,0]
        expected_state = [0,1,2,3,0,2,0,1,3,1]
        expected_reward= [0,1,2,3,1,3,0,2,4]
        correct_Q = np.array([
            [15.22977263322566,9.75298204],
            [15.38360872042996,5.96],
            [14.528897697404,8.841396],
            [12.6554522196,4.]
        ])

        s, info = env.reset()
        # this is a test for the environment, not rollout
        self.assertEqual(s, expected_state[0])

        a = action_plan[0]
        rollout.add_step_data(s, a)
        self.assertEqual(rollout.iter_ct, 1)

        for t in range(1, len(action_plan)):
            (s, r, term, trunc, info) = env.step(a)
            a = action_plan[t]
            rollout.add_step_data(s,a,r,term,trunc)

            # these are tests for the environment, not rollout
            self.assertEqual(s, expected_state[t])
            self.assertEqual(r, expected_reward[t-1])
            self.assertEqual(rollout.iter_ct, t+1)

        self.assertEqual(rollout.iter_ct, len(action_plan))
        (Q, Ind) = rollout.compute_enumerated_stateaction_value()

        self.assertEqual(Q.shape, correct_Q.shape)
        self.assertEqual(Ind.shape, correct_Q.shape)
        # we visited all
        self.assertTrue(np.all(Ind))
        # accurate Q function
        normalize_const = max(la.norm(correct_Q, ord=np.inf), la.norm(Q, ord=np.inf))
        self.assertLessEqual(la.norm(Q-correct_Q, ord=np.inf)/normalize_const, 1e-16)

    def test_compute_enumerated_stateaction_value_visit_partial(self):
        """ Same as test above but we do not have the last two
        steps/state-action values. We now have

          >> 9.6746631597       (0,0)*
          >> 9.772387029999999  (1,0)*
          >> 8.860997           (2,0)*
          >> 6.9303             (3,0)*
          >> 3.9699999999999998 (0,1)*
          >> 3.0000000000000004 (2,1)*
          >> 0.0                (0,0)
        """
        (env, rollout) = self.get_rollout_object()

        action_plan    = [0,0,0,0,1,1,0,1]
        expected_state = [0,1,2,3,0,2,0,1]
        expected_reward= [0,1,2,3,1,3,0]
        correct_Q = np.array([
            [9.6746631597,3.97],
            [9.77238703,0],
            [8.860997,3.],
            [6.9303,0]
        ])

        s, info = env.reset()
        # this is a test for the environment, not rollout
        self.assertEqual(s, expected_state[0])

        a = action_plan[0]
        rollout.add_step_data(s, a)
        self.assertEqual(rollout.iter_ct, 1)

        for t in range(1, len(action_plan)):
            (s, r, term, trunc, info) = env.step(a)
            a = action_plan[t]
            rollout.add_step_data(s,a,r,term,trunc)

            # these are tests for the environment, not rollout
            self.assertEqual(s, expected_state[t])
            self.assertEqual(r, expected_reward[t-1])
            self.assertEqual(rollout.iter_ct, t+1)

        self.assertEqual(rollout.iter_ct, len(action_plan))
        (Q, Ind) = rollout.compute_enumerated_stateaction_value()

        self.assertEqual(Q.shape, correct_Q.shape)
        self.assertEqual(Ind.shape, correct_Q.shape)
        # we visited all but two
        self.assertFalse(Ind[1,1])
        self.assertFalse(Ind[3,1])
        Ind[1,1] = Ind[3,1] = True
        self.assertTrue(np.all(Ind))
        # accurate Q function
        normalize_const = max(la.norm(correct_Q, ord=np.inf), la.norm(Q, ord=np.inf))
        self.assertLessEqual(la.norm(Q-correct_Q, ord=np.inf)/normalize_const, 2e-16)

    def test_compute_enumerated_stateaction_value_visit_multiepisode(self):
        """ 
        Combines last two tests to see check Rollout can combine multiple
        episodes to correctly estimate the Q function.
        """
        (env, rollout) = self.get_rollout_object()

        action_plan_1    = [0,0,0,0,1,1,0,1,1,0]
        expected_state_1 = [0,1,2,3,0,2,0,1,3,1]
        expected_reward_1= [0,1,2,3,1,3,0,2,4]
        correct_Q_1 = np.array([
            [15.22977263322566,9.75298204],
            [15.38360872042996,5.96],
            [14.528897697404,8.841396],
            [12.6554522196,4.]
        ])
        action_plan_2    = [0,0,0,0,1,1,0,1]
        expected_state_2 = [0,1,2,3,0,2,0,1]
        expected_reward_2= [0,1,2,3,1,3,0]
        correct_Q_2 = np.array([
            [9.6746631597,3.97],
            [9.77238703,0],
            [8.860997,3.],
            [6.9303,0]
        ])

        correct_avg_Q = (correct_Q_1 + correct_Q_2)/2.
        # double missing components since they were not visited in 2nd episode
        correct_avg_Q[1,1] *= 2
        correct_avg_Q[3,1] *= 2

        for i in range(2):
            action_plan = action_plan_1 if i == 0 else action_plan_2

            s, info = env.reset()

            a = action_plan[0]
            rollout.add_step_data(s, a)

            for t in range(1, len(action_plan)):
                (s, r, term, trunc, info) = env.step(a)
                a = action_plan[t]
                if t == len(action_plan)-1 and i == 0:
                    term = True
                rollout.add_step_data(s,a,r,term,trunc)

            # at end of first episode, spam some terminates and truncates
            if i == 0:
                for j in range(5):
                    a = env.action_space.sample()
                    (s, r, term, trunc, info) = env.step(a)
                    if j % 2 == 0:
                        trunc = True
                    else:
                        term = True
                    rollout.add_step_data(s,a,r,term,trunc)

        (Q, Ind) = rollout.compute_enumerated_stateaction_value()

        self.assertEqual(Q.shape, correct_avg_Q.shape)
        self.assertEqual(Ind.shape, correct_avg_Q.shape)
        # we visited all 
        self.assertTrue(np.all(Ind))
        # accurate Q function
        normalize_const = max(la.norm(correct_avg_Q, ord=np.inf), la.norm(Q, ord=np.inf))
        self.assertLessEqual(la.norm(Q-correct_avg_Q, ord=np.inf)/normalize_const, 1e-16)

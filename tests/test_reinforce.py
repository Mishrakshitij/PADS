"""Optional checks run when the TensorFlow training extra is installed."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pads import Environment, UserSimulator
from pads.policy import slot_filling_action
from pads.runner import run_episode


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow training extra is not installed")
class ReinforceTests(unittest.TestCase):
    def test_training_and_checkpoint_restore(self):
        from pads.reinforce import PolicyGradientREINFORCE
        agent = PolicyGradientREINFORCE(seed=5)
        restored = None
        try:
            env = Environment(UserSimulator(seed=5))
            for _ in range(3):
                result = run_episode(env, agent=agent, training=True)
                self.assertTrue(np.isfinite(result["loss"]))
            self.assertEqual(agent.session.run(agent.global_step), 3)
            state = env.reset()
            action, recurrent = agent.act(state, agent.initial_state(), env.action_mask(), greedy=True)
            self.assertTrue(env.action_mask()[action])
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = agent.save(directory)
                self.assertTrue(Path(checkpoint + ".index").is_file())
                restored = PolicyGradientREINFORCE(seed=99)
                restored.restore(directory)
                restored_action, restored_recurrent = restored.act(
                    state, restored.initial_state(), env.action_mask(), greedy=True
                )
                self.assertEqual(action, restored_action)
                for expected, actual in zip(recurrent, restored_recurrent):
                    np.testing.assert_allclose(actual, expected)
        finally:
            agent.close()
            if restored is not None:
                restored.close()

    def test_recurrent_rollout_matches_full_episode_forward_pass(self):
        from pads.reinforce import PolicyGradientREINFORCE
        agent = PolicyGradientREINFORCE(seed=11)
        try:
            env = Environment(UserSimulator(clean_prob=1, seed=11))
            state = env.reset()
            states = []
            recurrent = agent.initial_state()
            for _ in range(3):
                states.append(state.copy())
                _, recurrent = agent.act(state, recurrent, env.action_mask(), greedy=True)
                state, _, _ = env.step(slot_filling_action(state))
            initial_c, initial_h = agent.initial_state()
            whole_episode = agent.session.run(
                agent.next_recurrent,
                {agent.states: np.array(states)[None],
                 agent.initial_c: initial_c, agent.initial_h: initial_h},
            )
            for actual, expected in zip(recurrent, whole_episode):
                np.testing.assert_allclose(actual, expected, atol=1e-6)
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()

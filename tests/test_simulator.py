import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pads import Environment, UserSimulator
from pads.policy import Transition, discounted_returns, masked_probabilities, one_hot, select_action
from pads.politeness import DistilBertPoliteness, reward_from_labels, validate_classifier_labels
from pads.runner import run_episode
from pads.user import ACOUSTIC_FEATURES, AcousticSampler


class PolicyTests(unittest.TestCase):
    def test_one_hot_boundary_and_missing_label(self):
        for label in (None, -1, 8, 9):
            np.testing.assert_array_equal(one_hot(label, 8), np.zeros(8))
        self.assertEqual(one_hot(7, 8).tolist(), [0, 0, 0, 0, 0, 0, 0, 1])

    def test_masked_softmax_keeps_valid_zero_logits(self):
        np.testing.assert_allclose(masked_probabilities([0, 0, 100], [1, 1, 0]), [0.5, 0.5, 0])
        np.testing.assert_allclose(masked_probabilities([1000, 1000, -1000], [1, 1, 0]), [0.5, 0.5, 0])
        with self.assertRaises(ValueError):
            masked_probabilities([1, 2], [0, 0])

    def test_exploration_and_sampling_obey_mask(self):
        rng = np.random.default_rng(12)
        for exploration in (0.0, 0.5, 1.0):
            for _ in range(100):
                action = select_action([10, 0, 10, 0], [0, 1, 0, 1], rng, exploration=exploration)
                self.assertIn(action, (1, 3))

    def test_rollouts_own_state_and_mask_snapshots(self):
        state = np.arange(13, dtype=np.float32)
        mask = np.ones(8, dtype=bool)
        step = Transition.snapshot(state, 0, -1, mask)
        state[:] = 0
        mask[:] = False
        np.testing.assert_array_equal(step.state, np.arange(13))
        self.assertTrue(step.mask.all())
        self.assertFalse(step.state.flags.writeable)

    def test_discounted_returns(self):
        np.testing.assert_allclose(discounted_returns([-1, -1, 20], 0.9), [14.3, 17, 20])


class EnvironmentTests(unittest.TestCase):
    def test_reset_clears_previous_episode_and_returns_snapshot(self):
        env = Environment(UserSimulator(clean_prob=1, seed=4))
        state = env.reset()
        original = state.copy()
        env.step(0)
        np.testing.assert_array_equal(state, original)
        env.success = env.done = True
        env.acoustic_features = np.ones(8)
        reset = env.reset()
        self.assertFalse(env.done)
        self.assertFalse(env.success)
        self.assertIsNone(env.last_action)
        self.assertIsNone(env.acoustic_features)
        np.testing.assert_array_equal(reset[5:], np.zeros(8))

    def test_query_masks_match_available_and_uncovered_routes(self):
        env = Environment(UserSimulator(seed=1))
        env.reset()
        self.assertFalse(env.action_mask()[3:5].any())
        env.state[:5] = [1, 1, 1, 0, 0]
        self.assertEqual(env.action_mask()[3:5].tolist(), [True, False])
        mask = env.action_mask()
        env.state[:5] = [0, 1, 1, 1, 0]
        self.assertEqual(env.action_mask()[3:5].tolist(), [False, True])
        self.assertEqual(mask[3:5].tolist(), [True, False])

    def test_slot_policy_completes_noise_free_tasks(self):
        env = Environment(UserSimulator(clean_prob=1, seed=12))
        for _ in range(30):
            result = run_episode(env, policy="slot")
            self.assertEqual(result, {"reward": 18.0, "length": 3, "success": 1})
            with self.assertRaises(RuntimeError):
                env.step(0)

    def test_timeout_is_failure_and_invalid_query_is_terminal(self):
        env = Environment(UserSimulator(clean_prob=0, seed=2))
        self.assertEqual(run_episode(env, max_steps=2), {"reward": -2.0, "length": 2, "success": 0})
        env.reset()
        _, reward, done = env.step(3)
        self.assertEqual(reward, -10)
        self.assertTrue(done)
        self.assertFalse(env.success)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(8)

    def test_seeded_simulation_is_reproducible(self):
        def simulate():
            env = Environment(UserSimulator(seed=42))
            rng = np.random.default_rng(43)
            return [run_episode(env, policy="random", rng=rng) for _ in range(20)]
        self.assertEqual(simulate(), simulate())

    def test_runner_stores_state_before_the_selected_action(self):
        from pads.policy import slot_filling_action

        class RecordingAgent:
            def initial_state(self):
                self.inputs = []
                return 0

            def act(self, state, recurrent, mask, greedy=False):
                self.inputs.append(state.copy())
                return slot_filling_action(state), recurrent + 1

            def update(self, trajectory):
                self.trajectory = trajectory
                return 1.0

        agent = RecordingAgent()
        run_episode(Environment(UserSimulator(clean_prob=1, seed=2)), agent=agent, training=True)
        for state, step in zip(agent.inputs, agent.trajectory):
            np.testing.assert_array_equal(state, step.state)
        np.testing.assert_array_equal(agent.trajectory[0].state[5:], np.zeros(8))
        self.assertEqual(agent.trajectory[1].state[5 + agent.trajectory[0].action], 1)


class OptionalInputsTests(unittest.TestCase):
    def test_politeness_reward_table(self):
        expected = {(0, 0): -1.5, (1, 2): 2.5, (2, 1): -2.5, (3, 3): 5.0}
        for labels, reward in expected.items():
            self.assertEqual(reward_from_labels(*labels), reward)
        with self.assertRaises(ValueError):
            reward_from_labels(4, 0)

    def test_missing_classifier_has_actionable_error_without_importing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "checkpoint is not included"):
                DistilBertPoliteness(directory)

    def test_binary_classifier_is_rejected_before_optional_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({
                "id2label": {"0": "LABEL_0", "1": "LABEL_1"}
            }))
            with self.assertRaisesRegex(ValueError, "four-class checkpoint, found 2"):
                DistilBertPoliteness(directory)

    def test_declared_classifier_label_order_cannot_be_reversed(self):
        with self.assertRaisesRegex(ValueError, "id2label does not match"):
            validate_classifier_labels(4, {0: "polite", 1: "somewhat_polite",
                                          2: "somewhat_impolite", 3: "impolite"})
        validate_classifier_labels(4, {0: "impolite", 1: "somewhat_impolite",
                                       2: "somewhat_polite", 3: "polite"})

    def test_loaded_classifier_configuration_is_checked(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        classification = SimpleNamespace(ClassificationModel=Mock(return_value=SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(num_labels=2))
        )))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text('{"num_labels": 4}')
            with patch.dict(sys.modules, {"simpletransformers.classification": classification}):
                with self.assertRaisesRegex(ValueError, "four-class checkpoint, found 2"):
                    DistilBertPoliteness(directory)

    def test_acoustic_sampler_matches_history_and_returns_fixed_shape(self):
        from collections import Counter
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            row = {"myAction": 1, **{f"total_act{i}": 0 for i in range(8)},
                   **{name: index for index, name in enumerate(ACOUSTIC_FEATURES)}}
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            sampler = AcousticSampler(path)
            features, found = sampler.sample(1, Counter(), np.random.default_rng(0))
            self.assertTrue(found)
            np.testing.assert_array_equal(features, np.arange(8))
            features, found = sampler.sample(1, Counter({0: 1}), np.random.default_rng(0))
            self.assertFalse(found)
            np.testing.assert_array_equal(features, np.zeros(8))


class CliTests(unittest.TestCase):
    def test_cli_runs_outside_repository_and_writes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            source = str(Path(__file__).resolve().parents[1] / "src")
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source, environment.get("PYTHONPATH")]))
            result = subprocess.run(
                [sys.executable, "-m", "pads", "simulate", "--episodes", "3",
                 "--clean-prob", "1", "--output", "metrics.csv"],
                cwd=directory, env=environment, capture_output=True, text=True, check=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["success_rate"], 1)
            with (Path(directory) / "metrics.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)


if __name__ == "__main__":
    unittest.main()

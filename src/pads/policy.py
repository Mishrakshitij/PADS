"""Dependency-light action selection and episode bookkeeping."""

from dataclasses import dataclass

import numpy as np


def one_hot(label: int | None, size: int) -> np.ndarray:
    """Encode a valid zero-based label; missing/out-of-range labels are zero."""
    result = np.zeros(size, dtype=np.float32)
    if label is not None and 0 <= label < size:
        result[label] = 1
    return result


def masked_probabilities(logits, mask) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if logits.ndim != 1 or mask.shape != logits.shape or not mask.any():
        raise ValueError("Logits and mask must be equal-length vectors with a valid action.")
    if not np.isfinite(logits[mask]).all():
        raise ValueError("Valid action logits must be finite.")
    weights = np.zeros_like(logits)
    weights[mask] = np.exp(logits[mask] - logits[mask].max())
    return weights / weights.sum()


def select_action(logits, mask, rng, *, exploration=0.0, greedy=False) -> int:
    """Mask both stochastic policy sampling and optional uniform exploration."""
    if not 0 <= exploration <= 1:
        raise ValueError("exploration must be between 0 and 1")
    probabilities = masked_probabilities(logits, mask)
    if not greedy and exploration and rng.random() < exploration:
        return int(rng.choice(np.flatnonzero(mask)))
    if greedy:
        return int(probabilities.argmax())
    return int(rng.choice(len(probabilities), p=probabilities))


def discounted_returns(rewards, discount: float) -> np.ndarray:
    if not 0 <= discount <= 1:
        raise ValueError("discount must be between 0 and 1")
    result = np.zeros(len(rewards), dtype=np.float32)
    total = 0.0
    for index in reversed(range(len(rewards))):
        total = float(rewards[index]) + discount * total
        result[index] = total
    return result


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    mask: np.ndarray

    @classmethod
    def snapshot(cls, state, action, reward, mask):
        # The simulator and caller can safely reuse their arrays after a step.
        state_copy = np.array(state, dtype=np.float32, copy=True)
        mask_copy = np.array(mask, dtype=bool, copy=True)
        state_copy.flags.writeable = False
        mask_copy.flags.writeable = False
        return cls(state_copy, int(action), float(reward), mask_copy)


def slot_filling_action(state) -> int:
    """A deterministic sanity-check policy, not the learned PADS policy."""
    if not (state[1] or state[4]):
        return 0
    if not (state[0] or state[3]):
        return 1
    if not state[2]:
        return 2
    return 3 if state[0] and state[1] else 4

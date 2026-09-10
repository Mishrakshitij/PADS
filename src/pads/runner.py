"""Shared episode collection and CSV summaries for simulation and training."""

import csv
from pathlib import Path

import numpy as np

from .policy import Transition, slot_filling_action


def run_episode(env, *, max_steps=15, policy="slot", rng=None, agent=None, training=False):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if agent is None and policy not in ("slot", "random"):
        raise ValueError("policy must be 'slot' or 'random'")
    if training and agent is None:
        raise ValueError("training requires a policy-gradient agent")
    rng = rng if rng is not None else np.random.default_rng()
    state = env.reset()
    recurrent = agent.initial_state() if agent is not None else None
    trajectory = []
    total_reward = 0.0
    for length in range(1, max_steps + 1):
        mask = env.action_mask()
        if agent is not None:
            action, recurrent = agent.act(state, recurrent, mask, greedy=not training)
        elif policy == "slot":
            action = slot_filling_action(state)
        else:
            action = int(rng.choice(np.flatnonzero(mask)))
        next_state, reward, done = env.step(action)
        trajectory.append(Transition.snapshot(state, action, reward, mask))
        total_reward += reward
        state = next_state
        if done:
            break
    result = {"reward": total_reward, "length": length, "success": int(env.success)}
    if training:
        result["loss"] = agent.update(trajectory)
    return result


def summarize(rows):
    if not rows:
        raise ValueError("At least one episode is required.")
    return {
        "episodes": len(rows),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_length": float(np.mean([row["length"] for row in rows])),
        "success_rate": float(np.mean([row["success"] for row in rows])),
    }


def write_metrics(path, rows):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

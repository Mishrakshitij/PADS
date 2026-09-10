"""Slot-based bus-information simulator adapted from the original user.py."""

from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

ACOUSTIC_FEATURES = (
    "interruption", "interruption_total", "dtmf", "dtmf_total",
    "repeat", "repeat_total", "start_over", "start_over_total",
)


class AcousticSampler:
    """Look up features by the current action and preceding action counts."""

    def __init__(self, path: str | Path):
        self.rows = defaultdict(list)
        with Path(path).expanduser().open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"myAction", *ACOUSTIC_FEATURES, *(f"total_act{i}" for i in range(8))}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Simulator CSV is missing columns: {', '.join(sorted(missing))}")
            for row in reader:
                key = (int(row["myAction"]), *(int(row[f"total_act{i}"]) for i in range(8)))
                self.rows[key].append([float(row[name]) for name in ACOUSTIC_FEATURES])

    def sample(self, action: int, history: Counter, rng):
        candidates = self.rows.get((action, *(history[i] for i in range(8))))
        if not candidates:
            return np.zeros(8, dtype=np.float32), False
        return np.array(candidates[int(rng.integers(len(candidates)))], dtype=np.float32), True


class UserSimulator:
    def __init__(self, clean_prob=0.9, seed=None, sampler: AcousticSampler | None = None):
        if not 0 <= clean_prob <= 1:
            raise ValueError("clean_prob must be between 0 and 1")
        self.clean_prob = clean_prob
        self.rng = np.random.default_rng(seed)
        self.sampler = sampler
        self.history = Counter()
        self.goal = {}
        self.total_sample = self.not_in_sample = 0

    def _noise(self, response):
        return response if self.rng.random() < self.clean_prob else ">NOISE<"

    def reset(self):
        self.history.clear()
        self.total_sample = self.not_in_sample = 0
        self.goal = {
            "departure": str(self.rng.choice(["<d>", "<uncovered_d>"])),
            "arrival": str(self.rng.choice(["<a>", "<uncovered_a>"])),
            "time": "<time>",
        }
        return self._noise(str(self.rng.choice(list(self.goal.values()))))

    def respond(self, action, state):
        features = None
        if self.sampler is not None:
            features, found = self.sampler.sample(action, self.history, self.rng)
            self.total_sample += 1
            self.not_in_sample += int(not found)

        # Preserve the original simulator's history convention: repeated slot
        # requests return immediately and are not added to the action counts.
        if ((action in (0, 5) and (state[1] or state[4]))
                or (action in (1, 6) and (state[0] or state[3]))
                or (action in (2, 7) and state[2])):
            return "You already asked/know that!", features

        self.history[action] += 1
        requested_slot = {0: "arrival", 1: "departure", 2: "time",
                          5: "arrival", 6: "departure", 7: "time"}.get(action)
        if requested_slot:
            return self._noise(self.goal[requested_slot]), features
        queryable = (state[0] or state[3]) and (state[1] or state[4]) and state[2]
        covered = state[0] and state[1]
        if queryable and ((action == 3 and covered) or (action == 4 and not covered)):
            return "thank you", features
        return "That's not what I want!", features

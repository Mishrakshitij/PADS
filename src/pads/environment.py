"""PADS dialogue state and task/politeness rewards."""

import numpy as np

from .constants import ACTION_TEMPLATES, ENTITIES, NUM_ACTIONS, NUM_ENTITIES, STATE_DIM
from .policy import one_hot
from .user import UserSimulator


class Environment:
    """State: five observed slots followed by the previous action's one-hot code."""

    def __init__(self, user: UserSimulator, politeness_scorer=None):
        self.user = user
        self.politeness_scorer = politeness_scorer
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.done = False
        self.success = False
        self.last_action = None
        self.user_action = ""
        self.acoustic_features = None

    def reset(self):
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.done = self.success = False
        self.last_action = None
        self.acoustic_features = None
        self.user_action = self.user.reset()
        self._maintain_state()
        return self.state.copy()

    def _maintain_state(self):
        for index, entity in enumerate(ENTITIES):
            if entity in self.user_action:
                self.state[index] = 1
        self.state[NUM_ENTITIES:] = one_hot(self.last_action, NUM_ACTIONS)

    def queryable(self):
        return bool((self.state[0] or self.state[3]) and
                    (self.state[1] or self.state[4]) and self.state[2])

    def action_mask(self):
        # Slot-request actions remain legal after a slot is known, matching the
        # original action set. A repeated request still costs a turn.
        mask = np.ones(NUM_ACTIONS, dtype=bool)
        mask[3:5] = False
        if self.queryable():
            mask[3 if self.state[0] and self.state[1] else 4] = True
        return mask

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode has ended; call reset() before stepping again.")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"Action must be an integer from 0 to {NUM_ACTIONS - 1}.")
        self.last_action = int(action)
        self.user_action, self.acoustic_features = self.user.respond(action, self.state[:NUM_ENTITIES].copy())
        self._maintain_state()
        if self.user_action == "thank you" and self.queryable():
            self.done = self.success = True
            reward = 20.0
        elif self.user_action == "That's not what I want!":
            self.done, self.success = True, False
            reward = -10.0
        elif self.politeness_scorer is not None:
            reward = float(self.politeness_scorer(self.user_action, ACTION_TEMPLATES[action]))
        else:
            reward = -1.0
        return self.state.copy(), reward, self.done

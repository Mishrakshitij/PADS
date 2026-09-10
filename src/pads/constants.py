"""The eight actions and five slot indicators used by the archived runner."""

import json
from importlib.resources import files

NUM_ENTITIES = 5
NUM_ACTIONS = 8
STATE_DIM = NUM_ENTITIES + NUM_ACTIONS
ENTITIES = ("<d>", "<a>", "<time>", "<uncovered_d>", "<uncovered_a>")
ACTION_TEMPLATES = {
    int(key): value
    for key, value in json.loads(
        files("pads").joinpath("action_templates.json").read_text(encoding="utf-8")
    ).items()
}

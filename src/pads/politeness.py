"""Optional adapter for the classifier referenced, but not shipped, in RL_PADS.zip."""

import json
from pathlib import Path

import numpy as np

EXPECTED_LABELS = ("impolite", "somewhat_impolite", "somewhat_polite", "polite")


def validate_classifier_labels(num_labels, id_to_label=None):
    if num_labels != 4:
        raise ValueError(
            f"Politeness rewards require a four-class checkpoint, found {num_labels} labels. "
            "Expected 0=impolite, 1=somewhat_impolite, 2=somewhat_polite, 3=polite."
        )
    if id_to_label:
        actual = {int(key): str(value).strip().lower().replace("-", "_").replace(" ", "_")
                  for key, value in id_to_label.items()}
        for index, expected in enumerate(EXPECTED_LABELS):
            if actual.get(index) not in (expected, f"label_{index}", str(index)):
                raise ValueError(
                    "Checkpoint id2label does not match the required order: "
                    "0=impolite, 1=somewhat_impolite, 2=somewhat_polite, 3=polite. "
                    "Supply a checkpoint trained with this mapping."
                )


def reward_from_labels(user_label: int, bot_label: int) -> float:
    """Preserve the archive's four-class grouping and nonterminal reward table."""
    if user_label not in range(4) or bot_label not in range(4):
        raise ValueError("The original reward requires classifier labels 0, 1, 2, or 3.")
    return {(False, False): -1.5, (False, True): 2.5,
            (True, False): -2.5, (True, True): 5.0}[(user_label >= 2, bot_label >= 2)]


class DistilBertPoliteness:
    def __init__(self, checkpoint: str | Path):
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
            raise ValueError(
                f"No classifier checkpoint with config.json at {checkpoint}. "
                "The original outputs_dstc1 checkpoint is not included; supply a trained "
                "four-class SimpleTransformers DistilBERT checkpoint or use --reward task."
            )
        with (checkpoint / "config.json").open(encoding="utf-8") as handle:
            saved_config = json.load(handle)
        saved_labels = saved_config.get("id2label")
        saved_count = saved_config.get("num_labels", len(saved_labels) if saved_labels else None)
        if saved_count is not None:
            validate_classifier_labels(saved_count, saved_labels)
        try:
            from simpletransformers.classification import ClassificationModel
        except ImportError as exc:
            raise RuntimeError("Politeness rewards require: pip install -e '.[politeness]'") from exc
        self.model = ClassificationModel("distilbert", str(checkpoint), use_cuda=False)
        config = self.model.model.config
        validate_classifier_labels(config.num_labels, getattr(config, "id2label", None))

    def __call__(self, user_text: str, bot_text: str) -> float:
        predictions, _ = self.model.predict([user_text, bot_text])
        labels = np.asarray(predictions).reshape(-1)
        if len(labels) != 2:
            raise ValueError("Expected one class label for each utterance.")
        return reward_from_labels(int(labels[0]), int(labels[1]))

# Implementation and reproducibility notes

## Scope of the maintained code

The original code is preserved in [`archives/RL_PADS.zip`](../archives/RL_PADS.zip).
The maintained [`src/pads/`](../src/pads/) package is a documented revision of
its bus-information simulator and LSTM REINFORCE runner. It is not a verified
reproduction of the publication's six-domain experiments.

The manuscript describes six customer-support tasks, 10 or 12 actions, 25 or 30
turn limits, 500 evaluation dialogues, and AdaDelta. The archived runnable
prototype instead contains eight bus-information actions, five slot indicators,
a 15-turn limit, 400 evaluation dialogues, and RMSProp. The maintained runner
preserves that prototype's task and RMSProp optimizer. The CLI's lightweight
evaluation default is 100 dialogues and can be changed with `--eval-episodes`.
See [paper results](paper-results.md) for the published tables and their scope.

The six provided customer-support CSVs contain predicted politeness labels;
they are packaged as research data under [`datasets/`](../datasets/). This
revision does not invent a mapping from those records to the bus simulator's
state/action trajectories, or substitute them for missing classifier training
splits and domain environments.

## Source layout

| Archived source | Maintained location |
| --- | --- |
| `env.py` | `src/pads/environment.py`, optional classifier in `politeness.py` |
| `user.py` | `src/pads/user.py` |
| `util.py` | `src/pads/policy.py` |
| `rl/pg_reinforce.py` | `src/pads/reinforce.py`, trajectory helpers in `policy.py` |
| `run_mydata.py` | `src/pads/cli.py` and `runner.py` |
| Unused incomplete `hcn.py` | `legacy/hcn.py` |
| JSON templates and simulator CSVs | `data/simulator/`; active eight templates also packaged in `src/pads/` |
| Informal experiment notes and five-row result CSV | `legacy/` |

The installed package excludes the ZIP archive, corpus CSVs, and incomplete
legacy fragment. Repository data paths are explicit CLI arguments. The package
includes its active action templates and works outside the repository directory.

## Correctness changes

| Issue in the archive | Maintained behavior |
| --- | --- |
| Importing `env.py` loads a missing `outputs_dstc1` classifier. | Default task rewards need only NumPy; classifier loading occurs only for an explicitly requested local checkpoint. |
| User data uses a personal `~/Downloads/...` path; other paths concatenate the current directory. | `--sample-csv`, `--classifier-dir`, `--output`, and `--model-dir` select inputs and outputs explicitly. |
| `oneHotLabel(dim, dim)` indexes beyond the vector. | Out-of-range and missing labels return a zero vector. |
| Episode reset retains the prior last action and success flag. | Reset clears all episode state, including action, terminal status, success, and acoustic features. |
| Returned states and stored masks share mutable arrays. | Environment returns independent observations; trajectories snapshot immutable observations and masks. |
| Multiplying logits by zero does not remove illegal actions from softmax, and a custom softmax discards valid zero logits. | Masks explicitly remove prohibited actions; valid zero logits retain probability. |
| Uniform exploration can choose masked actions and skip recurrent updates. | All action selection respects masks. The maintained trainer samples from the masked policy and advances recurrent state every turn. |
| Rollouts save the recurrent state after the chosen action as its training input; per-turn updates use stale states. | Training recomputes the full observed episode from zero recurrent state, with the mask from each corresponding turn. |
| Return weighting also multiplies the regularizer, and the declared gradient limit is unused. | Cross-entropy is weighted by discounted returns; regularization is separate and global gradient norms are clipped at 5. |
| The return-history slice retains the oldest million entries forever. | A bounded queue retains the most recent million discounted returns. |
| Evaluation uses training exploration and a mutable rollout buffer. | Evaluation uses greedy legal actions, a separate seeded simulator, and no optimizer updates. |
| Saving uses the literal string `FLAGS.model_dir` and is commented out. | Training writes a TensorFlow checkpoint under the actual `--model-dir` and can restore a checkpoint directory or prefix. |

The training changes affect optimization and results. Stochastic policy sampling
replaces the archive's epsilon-greedy/argmax training; this matches the
return-weighted log-policy objective. Full-episode backpropagation replaces the
archive's one-step recurrent updates. Neither correction should be described as
an exact rerun of the original experiment.

## Simulator and reward semantics

State has 13 elements: departure, arrival, time, uncovered departure, uncovered
arrival, followed by the eight-element previous-action one-hot vector. Actions
0–2 request arrival/departure/time using the archive's polite templates; 5–7
request the same slots using its shorter templates. Action 3 returns a covered
route; action 4 reports no matching route. Query actions are enabled only when
the corresponding slots are known. Repeated slot requests remain selectable,
as in the archived runner.

The user's departure and arrival are independently sampled as covered or
uncovered. One random slot is volunteered at reset. Each volunteered or requested
slot is replaced with `>NOISE<` with probability `1 - clean_prob` (default 0.1).
The simulator's repeated-request history convention is preserved: those early
returns are not added to action counts.

Task rewards are +20 for successful completion, −10 for an invalid query, and
−1 per nonterminal turn. Reaching the turn limit without success is recorded as
failure, with no additional terminal penalty. `simulate --policy slot` is a
deterministic slot-filling sanity check; `--policy random` uniformly chooses a
legal action. Neither baseline represents the learned PADS model.

For `--reward politeness`, terminal rewards remain unchanged. Nonterminal rewards
preserve the archived four-class mapping:

| User class | System class | Reward |
| --- | --- | ---: |
| 0 or 1 | 0 or 1 | −1.5 |
| 0 or 1 | 2 or 3 | +2.5 |
| 2 or 3 | 0 or 1 | −2.5 |
| 2 or 3 | 2 or 3 | +5.0 |

The classifier receives the simulator's user response and the system action
template, matching the archived code. User responses may be symbolic slot/noise
tokens rather than natural utterances. Sampled acoustic features are exposed on
the environment but do not feed this reward. This adapter does not claim to
implement every manuscript reward variant (PRIS, PRSIS, or PRRIS).

## Running and checking the code

Core simulation supports Python 3.10+ and NumPy. The optional TensorFlow
compatibility runner targets Python 3.10 or 3.11 and TensorFlow 2.15.x; newer
Keras versions removed the legacy LSTM APIs used here.

```bash
python -m pip install -e .
python -m pads simulate --policy slot --episodes 100 --seed 42
python -m pads simulate --policy random --episodes 100 --seed 42 --output outputs/random.csv
python -m unittest discover -s tests -v

# Optional LSTM training in a Python 3.10/3.11 environment
python -m pip install -e '.[train]'
python -m pads train --episodes 1000 --eval-episodes 400 --seed 42 \
  --model-dir outputs/task-model --output outputs/task-training.csv
```

`simulate --output` contains one row per simulation episode. `train --output`
contains training episodes with a loss column; its printed JSON reports both
training and greedy evaluation summaries. Training uses a 32-unit LSTM,
learning rate 0.0001, discount 0.9, RMSProp decay 0.9, and L2 coefficient 0.001.
Learning rate, hidden size, discount, seed, turn limit, and clean probability
are CLI options. Discounted returns are normalized against the running return
history, preserving the archive's normalization strategy while fixing history
retention.

Classifier rewards require an additional installation and a supplied checkpoint:

```bash
python -m pip install -e '.[train,politeness]'
python -m pads train --reward politeness --classifier-dir /path/to/outputs_dstc1 \
  --episodes 1000 --seed 42 --model-dir outputs/polite-model
```

The archive does **not** include `outputs_dstc1`, classifier training code or a
verified classifier evaluation split. The optional classifier integration cannot
be validated against that missing model. A supplied model must use the original
four-class semantics; the adapter does not infer a different label mapping.
The required order is 0=impolite, 1=somewhat_impolite, 2=somewhat_polite,
3=polite. Both saved and loaded classifier configurations are checked for four
classes. Explicitly named `id2label` entries must match this order. Generic
`LABEL_0` through `LABEL_3` entries cannot establish class meaning, so the
checkpoint provider must verify their semantics before using this adapter.

Tests cover reset isolation, valid masking including zero logits and exploration,
immutable rollout storage, reward rules, deterministic seeded simulation, history
lookup, missing-checkpoint errors, and execution from another directory. Optional
TensorFlow tests check recurrent rollout/full-episode consistency, finite training
loss, and checkpoint save/restore. Such checks establish software behavior, not
published performance. TensorFlow results may vary across hardware/builds.
Restoring a policy checkpoint restores weights and optimizer state; the Python
return-normalization history and random-generator progress are not restored, so
`--restore` is not a bitwise continuation of a previous run.

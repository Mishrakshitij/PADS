# Release validation

The maintenance release was checked on Linux with Python 3.11.16,
NumPy 1.26.4, and TensorFlow CPU 2.15.1 in an isolated virtual environment.

| Check | Outcome |
| --- | --- |
| Build and install the package from `pyproject.toml` | Passed; packaged templates load outside the checkout. |
| Public test suite with TensorFlow installed, after a fresh package install | 35 tests passed, with no skips, including the format 3 dataset verifier tests. |
| Complete dataset verification against `RL_PADS_Data.zip` | All 10 CSV shards, source ZIP/member hashes, and every original field verified across 1,379,526 records. |
| Dataset verifier with an incorrect hash or reversed media part order | Both invalid releases rejected. |
| Four-class verifier regression cases | Source-field edits, altered source metadata, inconsistent counts, and invalid or fabricated blank-row labels are rejected. |
| Label coverage | 1,379,445 labeled records and 81 unscorable records across all six domains. |
| Dataset transfer into the repository | SHA-256 checked for every copied release file before replacement; original inputs retained locally and in Git history. |
| README dataset loading example | All three media shards load as 487,003 rows and 12 columns; 486,983 labeled rows and per-class counts match the manifest. |
| Slot and random simulator commands from outside the repository | Passed; JSON summaries and per-episode CSVs produced. |
| Five training episodes followed by four evaluation episodes | Passed; finite output and a TensorFlow checkpoint produced. |
| Recurrent rollout vs. full-episode computation; checkpoint save/restore | Covered by the passing TensorFlow regression tests. |
| Classifier configuration validation | Mock tests reject binary checkpoints, wrong named class order, and a mismatched loaded configuration. |
| Figure generation and visual inspection | Both README figures generated successfully from their recorded CSV sources. |
| Local Markdown targets and package metadata | All referenced local files exist; TOML parses. |

The baseline smoke commands use the default task reward, 15-turn limit,
clean-response probability 0.9, and seed 42:

```bash
python -m pads simulate --policy slot --episodes 100 --seed 42 --output runs/slot.csv
python -m pads simulate --policy random --episodes 100 --seed 42 --output runs/random.csv
python -m pads train --episodes 5 --eval-episodes 4 --seed 42 \
  --model-dir checkpoints/smoke --output runs/smoke.csv
```

For the two simulator checks, the slot policy had mean reward 17.62, mean length
3.38, and success rate 1.00; the random policy had mean reward 4.41, mean length
10.50, and success rate 0.71. These are software smoke tests on the small
bus-scheduling simulator. They are not the learned model's published benchmark
results and should not be compared with the six-domain manuscript table.

The original DistilBERT politeness checkpoint was not supplied, so the
prototype's DistilBERT reward path and end-to-end politeness training were not
exercised. The six-domain
paper results were transcribed and cross-checked against manuscript tables,
not reproduced. TensorFlow 2.15 emits expected deprecation notices for its legacy
recurrent APIs. Manuscript checks and the absence of a full LaTeX build are
recorded in [overleaf-maintenance.md](overleaf-maintenance.md).

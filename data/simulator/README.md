# Bus-dialogue simulator assets

These files were extracted without changing their contents from
[`archives/RL_PADS.zip`](../../archives/RL_PADS.zip). They support the archived
bus-information prototype, independently of the six customer-support CSVs in
[`datasets/`](../../datasets/).

| File | Original archive member | Contents |
| --- | --- | --- |
| `sample_from.csv` | `RL_PADS/data/sample_from.csv` | 1,918 rows, 32 columns |
| `sample_from_action_567_addition.csv` | `RL_PADS/sample_from_action_567_addition.csv` | 15,344 rows, 30 columns |
| `action_templates.json` | `RL_PADS/data/actionID_to_template.json` | 9 templates, IDs 0–8 |
| `action_templates_extended.json` | `RL_PADS/actionID_to_template.json` | 11 templates, IDs 0–10 |

The archive's root-level `RL_PADS/sample_from.csv` is byte-identical to
`RL_PADS/data/sample_from.csv`; one copy is extracted here. Both distinct JSON
template versions are retained. The runnable simulator uses the first eight
templates from `action_templates.json`, which match the original eight-action
runner. A packaged copy of these eight templates lets the CLI run from any
working directory.

The CSVs contain bus-dialogue records and acoustic/interaction indicators.
`AcousticSampler` matches `myAction` and `total_act0` through `total_act7`, then
returns `interruption`, `interruption_total`, `dtmf`, `dtmf_total`, `repeat`,
`repeat_total`, `start_over`, and `start_over_total`. Missing histories return
eight zeros, matching the archive. The sampler returns features for inspection;
the archived reward function did not consume them, and the maintained simulator
does not use them to claim acoustic reward adaptation.

To exercise this optional lookup from the repository root:

```bash
python -m pads simulate --policy random --episodes 100 --seed 42 \
  --sample-csv data/simulator/sample_from_action_567_addition.csv
```

These are historical input assets, not newly collected dialogue records or a
verified copy of the paper's six-domain training/evaluation split. Source data
permissions are not expanded by this repository's reorganization.

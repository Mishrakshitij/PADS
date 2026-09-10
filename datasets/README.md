# PADS dataset

PADS covers customer–agent conversations in six domains: airline, fastfood, finance, insurance, media, and software. The source corpus contains **1,379,526 utterance rows**.

The complete release contains **1,379,445 labeled rows** and **81 unscorable rows**, stored in **10 CSV shards**.

This release adds four-class labels to the source corpus using the categories defined in the [2022 PADS paper](https://doi.org/10.1016/j.neucom.2022.04.029). The [annotation guidelines](../docs/annotation-guidelines.md) explain the categories and examples. The [manifest](manifest.json) records source files, coverage, and checksums. The original utterance fields and binary predictions are retained.

| Label | Politeness class |
| --- | --- |
| `0` | `impolite` |
| `1` | `somewhat_impolite` |
| `2` | `somewhat_polite` |
| `3` | `polite` |

The scale measures expressed courtesy. Label `0` includes ordinary direct questions and bare slot values; it does not by itself imply hostility or dissatisfaction. The guidelines include the paper's examples.

## Coverage and distribution

| Domain | Labeled rows | Unscorable | 0 | 1 | 2 | 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| airline | 237,515 | 21 | 86,797 (36.5%) | 4,208 (1.8%) | 89,516 (37.7%) | 56,994 (24.0%) |
| fastfood | 181,129 | 27 | 77,556 (42.8%) | 5,267 (2.9%) | 76,963 (42.5%) | 21,343 (11.8%) |
| finance | 157,622 | 3 | 62,073 (39.4%) | 2,545 (1.6%) | 66,143 (42.0%) | 26,861 (17.0%) |
| insurance | 229,975 | 7 | 86,869 (37.8%) | 2,176 (0.9%) | 90,320 (39.3%) | 50,610 (22.0%) |
| media | 486,983 | 20 | 183,302 (37.6%) | 5,157 (1.1%) | 178,999 (36.8%) | 119,525 (24.5%) |
| software | 86,221 | 3 | 33,431 (38.8%) | 844 (1.0%) | 30,292 (35.1%) | 21,654 (25.1%) |
| **Total** | **1,379,445** | **81** | **530,028 (38.4%)** | **20,197 (1.5%)** | **532,233 (38.6%)** | **296,987 (21.5%)** |

The source inventory contains 81 whitespace-only text rows: airline 21, fastfood 27, finance 3, insurance 7, media 20, and software 3. These rows retain missing labels with `annotation_status=unscorable_empty_text` and are excluded from class percentages.

[`summary.csv`](summary.csv) provides machine-readable class counts; [`manifest.json`](manifest.json) records per-domain coverage and file hashes.

## Files and schema

Read each domain's CSV shards in the order listed under `domains[].files` in the manifest. Shards repeat the header and are at most 48 MiB. Shard boundaries serve file storage and may fall within a conversation.

Every CSV has the following header:

```text
,id,prediction,pred_score,conversation_id,root,speaker,text,turnNumber,politeness_label,politeness_label_name,annotation_status
```

| Column | Content |
| --- | --- |
| unnamed first column | Original exported row index. |
| `id` | Original utterance identifier; it may repeat. |
| `prediction`, `pred_score` | Original binary output and score retained from the supplied archive. Use `politeness_label` for the four-class task. |
| `conversation_id`, `root` | Original conversation and root identifiers. |
| `speaker` | Original speaker value, usually `agent` or `customer`. |
| `text` | Original utterance text. |
| `turnNumber` | Original turn position, retained as text. |
| `politeness_label` | `0`, `1`, `2`, or `3` for annotated utterances; empty for unscorable text. |
| `politeness_label_name` | Corresponding class name; empty for unscorable text. |
| `annotation_status` | `annotated` or `unscorable_empty_text`. |

The nine original fields and source row order are preserved. Existing malformed metadata and repeated identifiers therefore remain visible. In particular, source records include noninteger turn positions and missing speaker values. Inspect these fields before reconstructing conversations or defining train/test splits.

## Loading a domain

This example uses the optional `pandas` package. Loading strings preserves identifiers and missing labels:

```python
import json
from pathlib import Path

import pandas as pd

dataset_dir = Path("datasets")
manifest = json.loads((dataset_dir / "manifest.json").read_text())
domain = next(item for item in manifest["domains"] if item["domain"] == "media")
data = pd.concat(
    [pd.read_csv(dataset_dir / shard["path"], dtype=str, keep_default_na=False)
     for shard in domain["files"]],
    ignore_index=True,
)
annotated = data.loc[data["annotation_status"] == "annotated"].copy()
annotated["politeness_label"] = annotated["politeness_label"].astype("int8")
```

For large workflows, iterate over each shard with `pd.read_csv(..., chunksize=100_000)`. Define evaluation splits at the conversation level after checking identifier quality; the release does not designate train/test partitions.

## Integrity and source data

Verify the release with the Python standard library:

```bash
python scripts/verify_dataset.py
```

If the original source ZIP is available, also check every exported row's original nine fields against it:

```bash
python scripts/verify_dataset.py --source-archive RL_PADS_Data.zip
```

The release includes the original source manifest. `manifest.json` records the release date, label definitions, source archive's identity, coverage, and file hashes.

The corpus is based on [MultiDoGO](https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset). Its Community Data License Agreement – Permissive, Version 1.0 [license](../licenses/MultiDoGO-LICENSE.txt) and [notice](../licenses/MultiDoGO-NOTICE.txt) are retained. Source utterances include account-, contact-, and address-like strings as originally supplied; authenticity has not been established.

For the paper's BibTeX entry, see the [repository README](../README.md).

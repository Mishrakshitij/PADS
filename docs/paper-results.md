# Paper reference and manuscript-reported results

Kshitij Mishra, Mauajama Firdaus, and Asif Ekbal. **Please be polite: Towards building a politeness adaptive dialogue system for goal-oriented conversations.** *Neurocomputing*, volume 494, pages 242–254, 2022. The publisher lists the issue date as **14 July 2022**. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0925231222003952) · [DOI](https://doi.org/10.1016/j.neucom.2022.04.029).

The bibliographic fields were checked against the publisher's indexed record and the [Crossref BibTeX export](https://api.crossref.org/works/10.1016/j.neucom.2022.04.029/transform/application/x-bibtex). The author's [GitHub profile](https://github.com/Mishrakshitij) links to [Google Scholar](https://scholar.google.com/citations?user=jfTVBUQAAAAJ); Scholar was rate-limited during verification. PADS is the 2022 Neurocomputing work; **GenPADS is a separate 2023 paper**.

The numbers below were transcribed from the author-provided Overleaf project, file `elsarticle-template.tex`, commit **`e34cb6547dbc119956c2c3fff0b604766d6b76d6`**. They are **manuscript-reported results**, not results reproduced by running this repository. The publisher's full text could not be fetched to establish that every table matches the final typeset article. Table references below are the literal LaTeX labels, which remain identifiable independently of typeset table numbering.

## Task and reward variants

The manuscript describes a dialogue manager trained with REINFORCE and a 32-unit LSTM. A DistilBERT classifier assigns four politeness levels to utterances: `0 = impolite`, `1 = somewhat_impolite`, `2 = somewhat_polite`, and `3 = polite`. The agent uses these labels in rewards while a simulated user supplies task information.

| Variant | Manuscript description | LaTeX label |
| --- | --- | --- |
| Baseline | Rewards task success or failure and penalizes additional turns. | `alg:Reward1` |
| PRIS | Adds politeness feedback for sampled task-intent utterances. | `alg:Reward2` |
| PRSIS | Extends PRIS with politeness and noise feedback on slot-value responses. | `alg:Reward3` |
| PRRIS | Extends PRSIS with a reward of −2.5 for repeated agent actions. | `alg:Reward4` |

These are the manuscript's algorithm names. The released bus-scheduling prototype has its own reward switches and should not be presented as a complete implementation of all six-domain experiments.

## Task success rate

Source: `Table:5` and the matching `Temp+Ret` rows of `user_sim_eval`. Values are success-rate fractions; higher is better. The `Table:5` caption and adjacent prose specify **15,000 training dialogues**. The implementation section separately says **6,000 episodes**; that discrepancy remains unresolved.

| Domain | Baseline | PRIS | PRSIS | PRRIS |
| --- | ---: | ---: | ---: | ---: |
| airline | 0.554 | 0.600 | **0.658** | 0.632 |
| fastfood | 0.572 | 0.608 | 0.684 | **0.782** |
| finance | 0.602 | 0.650 | 0.712 | **0.714** |
| insurance | 0.664 | 0.666 | 0.706 | **0.724** |
| media | 0.566 | 0.608 | 0.654 | **0.732** |
| software | 0.488 | 0.546 | **0.632** | 0.616 |

The machine-readable source for the repository chart is [paper-results.csv](paper-results.csv). PRRIS has the highest tabulated rate in four domains; PRSIS leads in airline and software. No confidence intervals or run-to-run deviations are provided in this table.

## Politeness classification

Source: `Table:3`. The manuscript describes these as weighted averages over four classes. F1 is the F1 score, GM is geometric mean, and IBA is index-balanced accuracy. Values are retained as printed, without recomputing them from the released annotations.

| Domain | BiLSTM F1 | BiLSTM GM | BiLSTM IBA | DistilBERT F1 | DistilBERT GM | DistilBERT IBA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| airline | 0.90 | 0.92 | 0.85 | 0.92 | 0.94 | 0.88 |
| fastfood | 0.86 | 0.90 | 0.81 | 0.89 | 0.92 | 0.85 |
| finance | 0.90 | 0.92 | 0.85 | 0.92 | 0.94 | 0.88 |
| insurance | 0.92 | 0.93 | 0.87 | 0.93 | 0.94 | 0.89 |
| media | 0.91 | 0.93 | 0.86 | 0.94 | 0.95 | 0.91 |
| software | 0.90 | 0.92 | 0.85 | 0.93 | 0.95 | 0.89 |
| All Domains | 0.86 | 0.89 | 0.81 | 0.90 | 0.92 | 0.87 |

The weighted summary favors DistilBERT in every row. This does not mean that DistilBERT wins every individual class or metric in the separate per-class table, `Table:2`.

## Manuscript data statistics

Source: `Table:1`. These are manuscript counts, not an inventory of the files supplied in this release. Phase 1 represents original instances processed during annotation; Phase 2 represents the manually verified gold-standard subset. The stated splits concern classifier experiments.

| Domain | Phase 1 | Phase 2 | Train | Test | Validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| airline | 237,514 | 10,214 | 7,149 | 1,992 | 1,073 |
| fastfood | 173,406 | 10,058 | 7,040 | 1,961 | 1,057 |
| finance | 157,229 | 10,065 | 7,044 | 1,962 | 1,059 |
| insurance | 229,741 | 10,108 | 7,075 | 1,971 | 1,062 |
| media | 345,925 | 10,378 | 7,264 | 2,024 | 1,090 |
| software | 86,220 | 10,347 | 7,242 | 2,018 | 1,087 |

## Simulated tasks

Source: `Table:4`. Each domain uses a single task. The table's slot-value count denotes groups of information requested by the simulator.

| Domain | Task | Slot-value groups | Agent actions |
| --- | --- | ---: | ---: |
| airline | `book_Flight` | 4 | 12 |
| fastfood | `order_Pizza` | 4 | 12 |
| finance | `check_Balance` | 3 | 10 |
| insurance | `check_ClaimStatus` | 3 | 10 |
| media | `start_Service` | 3 | 10 |
| software | `start_Order` | 4 | 12 |

## User simulator comparison

Source: `user_sim_eval`. `Temp` uses hand-authored templates for responses; `Temp+Ret` retrieves the user's task-intent utterance from a dataset and uses templates for slot information. These are simulated task-completion rates, not measurements from deployed user conversations.

| Domain | Simulator | Baseline SR | PRIS SR | PRSIS SR | PRRIS SR |
| --- | --- | ---: | ---: | ---: | ---: |
| airline | Temp | 0.579 | 0.564 | 0.621 | 0.637 |
| airline | Temp+Ret | 0.554 | 0.600 | 0.658 | 0.632 |
| fastfood | Temp | 0.575 | 0.569 | 0.629 | 0.674 |
| fastfood | Temp+Ret | 0.572 | 0.608 | 0.684 | 0.782 |
| finance | Temp | 0.611 | 0.594 | 0.664 | 0.683 |
| finance | Temp+Ret | 0.602 | 0.650 | 0.712 | 0.714 |
| insurance | Temp | 0.658 | 0.674 | 0.691 | 0.679 |
| insurance | Temp+Ret | 0.664 | 0.666 | 0.706 | 0.724 |
| media | Temp | 0.581 | 0.577 | 0.667 | 0.638 |
| media | Temp+Ret | 0.566 | 0.608 | 0.654 | 0.732 |
| software | Temp | 0.520 | 0.535 | 0.584 | 0.610 |
| software | Temp+Ret | 0.488 | 0.546 | 0.632 | 0.616 |

The same source reports METEOR scores for the similarity of user responses, which should not be confused with task success. Every `Temp` METEOR entry is `0.999`; the `Temp+Ret` entries are:

| Domain | Baseline MET | PRIS MET | PRSIS MET | PRRIS MET |
| --- | ---: | ---: | ---: | ---: |
| airline | 0.516 | 0.613 | 0.582 | 0.574 |
| fastfood | 0.443 | 0.421 | 0.498 | 0.475 |
| finance | 0.541 | 0.506 | 0.584 | 0.527 |
| insurance | 0.581 | 0.563 | 0.538 | 0.594 |
| media | 0.452 | 0.507 | 0.488 | 0.516 |
| software | 0.444 | 0.482 | 0.436 | 0.493 |

## Sequence-to-sequence baseline

Source: `SDG_eval`. The sequence-to-sequence dialogue generator (SDG) is described as a fine-tuned BART model, separate from the RL policy. PPL is perplexity, MET is METEOR, F1 is described as ROUGE-2 F1, and APL is the mean proportion of polite utterances in a dialogue. The manuscript describes approximately 1,000 training and 250 test dialogues per domain. These table values also appear in the authors' [IIT Patna IJCAI 2023 tutorial](https://www.iitp.ac.in/~ai-nlp-ml/resources/talks/IJCAI23-Tutorial-Final.pdf).

| Domain | PPL | BLEU | MET | F1 | APL |
| --- | ---: | ---: | ---: | ---: | ---: |
| airline | 1.598 | 0.051 | 0.340 | 0.212 | 0.772 |
| fastfood | 1.973 | 0.010 | 0.265 | 0.151 | 0.748 |
| finance | 1.907 | 0.045 | 0.265 | 0.161 | 0.787 |
| insurance | 2.031 | 0.041 | 0.331 | 0.205 | 0.790 |
| media | 1.576 | 0.038 | 0.326 | 0.189 | 0.752 |
| software | 1.929 | 0.038 | 0.332 | 0.181 | 0.746 |

## Human evaluation

Source: `Table:6` and the Human Evaluation subsection. The manuscript describes 150 dialogues evaluated by six experts, with politeness adaptability (PA) and task completion rate (TCR) rated on a five-point scale. It reports Fleiss' kappa of 0.63 for PA and 0.59 for TCR.

| Model | PA, as printed | TCR, as printed |
| --- | ---: | ---: |
| Baseline | 0.35 | 0.41 |
| PRIS | 0.38 | 0.45 |
| PRSIS | 0.43 | 0.49 |
| PRRIS | 0.47 | 0.53 |

**Scale clarification is needed:** the printed table values are below 1, whereas the text specifies a 1–5 rating scale. The source does not explain a normalization. These values are preserved verbatim and should not be relabeled as percentages, normalized scores, or raw five-point means.

## Reproduction boundaries and source ambiguities

The original `RL_PADS.zip` is a bus-scheduling prototype. Its training script has eight actions, five entity indicators, a 15-turn limit, 400 evaluation dialogues, and an RMSProp optimizer. The manuscript describes six other domain tasks, 10 or 12 actions, 25- or 30-turn limits, 500 evaluation dialogues, and AdaDelta. The archive also points to a local classifier checkpoint and local sampling file; it does not contain the trained politeness classifier, classifier training program, or the complete domain-specific simulator implementations.

The archived `All_experiments_details` contains 100-episode pilot logs. It is not the source of the manuscript success-rate table and cannot establish reproduction of that table. Fixes that make the released prototype portable or correct its runtime behavior should be recorded as repository maintenance, rather than evidence of reproducing the paper.

The source has several unresolved differences that affect reconstruction:

- The success-rate table specifies 15,000 training dialogues, while Implementation Details specifies 6,000 episodes.
- The User Simulator subsection gives a polite-action noise probability of 0.05. The PRSIS subsection instead gives a polite-action clean-response probability of 0.9, implying noise probability 0.1.
- The human-evaluation rating scale and printed table values do not have a documented conversion.
- The prose states that DistilBERT improves every class and metric, but the per-class `Table:2` includes exceptions. The summary above restricts that claim to the weighted averages in `Table:3`.

Resolving these questions requires the original experiment configurations, checkpoints, or evaluation records. No manuscript numbers have been changed to match the prototype or newly generated repository statistics.

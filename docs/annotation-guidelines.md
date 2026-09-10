# Annotation guidelines

These guidelines summarize the four politeness categories described in the [2022 PADS paper](https://doi.org/10.1016/j.neucom.2022.04.029). They concern individual utterances in customer–agent conversations across airline, fastfood, finance, insurance, media, and software.

## Categories and examples

Assign one category to each utterance, using the following examples from the paper as reference points.

| Label | Category | Distinction illustrated by the example | Paper example |
| --- | --- | --- | --- |
| `0` | `impolite` | Direct wording without an expressed courtesy cue. | “What is your departure city?” |
| `1` | `somewhat_impolite` | A briefly softened request. | “Can you provide your departure city?” |
| `2` | `somewhat_polite` | Clear courtesy or consideration. | “Please, provide your departure city?” |
| `3` | `polite` | More elaborated, considerate wording. | “For further processing, I would like to know your departure city, please, name it?” |

Label `0` includes ordinary direct questions. It should therefore be read as a position on this courtesy scale, rather than as a claim that every such utterance is hostile.

## Applying the categories

- **Consider the speaker and task context.** The paper labels the customer's “Can you book my flight ticket?” as `1`, and the agent's “Can you provide your booking confirmation number?” as `2`. Similar request forms can receive different labels. Read these alongside the departure-city examples when judging an utterance.
- **Attend to courtesy in the wording.** The paper highlights “No problem! Please try uninstalling and reinstalling the app again.”, “Okay! Please attach a copy of your receipt?”, and “Please solve this problem.” as cases where an earlier binary classifier missed politeness. Review such cues when choosing among the four categories.
- **Distinguish slot information from dissatisfaction.** A short answer supplying a city, date, or other requested value serves a task function. The paper explicitly excludes slot-value responses when interpreting low user politeness as dissatisfaction.
- **Keep four-class judgments separate from binary scores.** The paper introduces the four categories because a polite/impolite decision alone loses distinctions needed for dialogue adaptation.

Source: **Rationale for Polite Orientation** and **Politeness Annotation**, within the paper's **Preparing Politeness Oriented Conversational Data** section.

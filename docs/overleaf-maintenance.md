# Manuscript maintenance

The author-provided Overleaf project was reviewed against source commit
`e34cb6547dbc119956c2c3fff0b604766d6b76d6`. The repository's
[reported result tables](paper-results.md) retain that commit as their source.

Maintenance of `elsarticle-template.tex` makes these limited changes:

- Replace the leftover template journal name with `Neurocomputing`.
- Remove the unused title note advertising the Elsevier LaTeX template.
- Correct the case of `PADS_Diagram.PNG` to match the actual
  `PADS_Diagram.png` asset on case-sensitive filesystems.
- Add a code and data availability section linking to this repository and its
  explanation of the released implementation's scope.
- Correct `Kshtij Mishra` to `Kshitij Mishra` in the acknowledgement.

No experimental results, mathematical definitions, or scientific claims were
changed. Conflicting episode counts and other research ambiguities are recorded
in the [paper notes](paper-results.md) for the authors to resolve from original
experiment records.

Validation checked all 14 active image references and found all files present;
the missing image in a commented-out figure remains inactive. The Git diff
passed whitespace checks. A TeX engine was not available in the maintenance
environment, so a full PDF compilation was not performed and the existing PDF
snapshot was not regenerated. Overleaf remains the manuscript's source of truth;
the full manuscript is not duplicated in this repository.

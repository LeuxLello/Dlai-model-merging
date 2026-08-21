# Example-level error analysis: interpretation

## Scope

This is a descriptive seed-42 analysis, not a new confirmatory test. Mean and frozen TIES were
evaluated on up to 500 real validation examples per task for all six task pairs. Each prediction was
classified as specialist competence preserved, merge-induced loss, merge recovery, or shared
failure.

The analyzed data comprise 500 sampled SST-2 validation sentences, 500 sampled IMDb test reviews,
all 408 MRPC validation pairs, and all 277 RTE validation pairs. The complete Kaggle artifact holds
10,110 example-method rows. Raw texts and representative examples remain in that artifact rather
than GitHub to avoid redistributing dataset content.

## Main patterns

- The compatible SST-2+IMDb merge was comparatively stable. TIES changed accuracy by -1.4
  percentage points on IMDb and -3.4 on SST-2 relative to the corresponding specialists.
- IMDb+RTE was fragile and asymmetric. TIES lost 13.6 points on IMDb and 8.7 on RTE.
- The largest single TIES loss was 12.8 points on the SST-2 side of SST-2+RTE.
- Transfer was not exclusively negative: Mean improved RTE by 1.44 points when merged with SST-2,
  even though the same merge strongly damaged SST-2.
- Across the 12 task-pair comparisons, TIES achieved higher merged accuracy than Mean in 8 cases.

Across the repeated example evaluations, Mean produced 722 merge losses and 426 merge gains. TIES
reduced these to 663 losses and 409 gains, preserving 59 additional correct specialist decisions.
This supports the interpretation that TIES is usually more conservative, not universally superior.

## Scientific interpretation

Compatibility is directional at the task level: one task can benefit while its partner degrades.
Pair-average retention therefore hides meaningful asymmetry. The real-example transitions explain
why parameter-space alignment is predictive but imperfect: merging can change individual decisions
in both directions, and the net score is the balance between recovered and destroyed competence.

These records also provide the natural bridge to a visual demonstration. A future interface can
present one input, the specialist decision, the merged decision, confidence, and transition type,
allowing users to see model fusion as a concrete exchange of competences rather than only a table of
aggregate metrics.

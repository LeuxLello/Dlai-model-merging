# Layer-adaptive improvement: interpretation

## Protocol

Seed 42 was used only for development. For adaptive Mean and adaptive TIES, one global early-layer
factor was selected from 0.25, 0.50, 0.75, and 1.00 using all six task pairs. Embeddings and late
layers remained at factor 1.0. The selected factor was then frozen before evaluation on held-out
seeds 7 and 123.

## Result

Both methods selected factor **1.00**, the explicit no-adaptation control. Development retention
increased monotonically as the factor approached 1.00:

- adaptive Mean: 0.8875, 0.9060, 0.9203, 0.9296;
- adaptive TIES: 0.8895, 0.9105, 0.9214, 0.9332.

Consequently, adaptive and standard methods were identical on all 12 held-out pair-seed units. The
paired mean delta, median delta, and bootstrap confidence interval were all exactly zero. This is a
valid negative result: uniformly shrinking the merged update in BERT layers 0-1 does not improve
Mean or TIES under the declared protocol.

## What remains supported

On held-out seeds 7 and 123, standard TIES retained 0.9368 of specialist performance on average,
compared with 0.9254 for Mean, a difference of approximately 1.15 percentage points. TIES exceeded
Mean on 9 of 12 pair-seed units. This comparison is descriptive here because the experiment's
pre-declared primary comparison was adaptive versus its corresponding standard method.

## Scientific conclusion

The earlier layer-wise ablation identified cumulative interference and a relatively damaging early
block contribution, but that observation does not imply that uniform early-layer shrinkage is a
useful remedy. The combined evidence instead points toward selective parameter-level conflict:
future improvement experiments should preserve compatible early-layer updates while suppressing
only coordinates or substructures that show destructive disagreement.

# Scope-density TIES: interpretation

## Protocol

Seed 42 selected one global TIES density schedule across all six task pairs. The candidates were
uniform `(0.20, 0.20, 0.20)`, early-sparse `(0.20, 0.10, 0.20)`, early-very-sparse
`(0.20, 0.05, 0.20)`, and depth-progressive `(0.10, 0.10, 0.30)`, ordered as embeddings, early,
and late scopes. The winner was frozen before evaluation on seeds 7 and 123.

## Result

Uniform density was the strongest development schedule:

- uniform: 0.9332 mean retention;
- depth-progressive: 0.9278;
- early-sparse: 0.9182;
- early-very-sparse: 0.9038.

Because the selected schedule was the explicit TIES control, Scope-TIES and standard TIES were
identical on all 12 held-out pair-seed units. Their paired delta and bootstrap interval were exactly
zero. Standard TIES remains the strongest frozen method with 0.9368 mean held-out retention,
compared with 0.9254 for Mean.

## Conclusion

The experiment rules out a second simple remedy: neither uniform early-layer scaling nor a coarse
three-scope trimming-density schedule improves TIES under the declared protocol. Further schedule
search would increase researcher degrees of freedom without strong evidence. The next analysis
should instead inspect example-level prediction transitions to determine which real inputs are
preserved, lost, or recovered after merging.

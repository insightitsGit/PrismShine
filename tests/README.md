# PrismShine tests

Testing strategy: docs/DESIGN.md §11.

Planned layout:

- `fixtures/bundles/` — canonical EvidenceBundle fixtures: one healthy baseline plus one firing + one non-firing fixture per Handbook signature (HANDBOOK.md §4 rule 1)
- `test_handbook_*.py` — per-detector-family signature tests
- `test_copycheck.py` — fact extraction/normalization table tests + arithmetic-closure cases (derived figures must not flag)
- `test_coverage.py` — vector coverage math (incl. JL-fallback, resonance mode, and composite support for cross-chunk synthesis)
- `test_contradiction.py` — negation-asymmetry/opposite-verb cases: contradicted-but-similar sentences must be promoted to Tier 3, near-identical grounded pairs must NOT fire
- `test_fusion.py` — weight/band/gate-naming tests
- `test_golden_verdicts.py` — bundle -> ShineVerdict snapshot determinism
- `test_capabilities.py` — degradation matrix (DESIGN §8.2): for each absent optional dependency, gate builds, verdicts record the degraded mode, gray zones resolve to flag (never pass), user-supplied Embedder/Judge substitution works
- `test_integration_chorusgraph.py` — demo graph with injected failures

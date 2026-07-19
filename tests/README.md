# PrismShine tests

| Suite | Location |
|---|---|
| Models / hash / builder | `test_models.py` |
| Import purity | `test_import_purity.py` |
| Handbook detectors | `forensics/test_detectors.py` |
| Copy-check / coverage / fusion | `test_copycheck.py`, `test_coverage.py`, `test_fusion_*.py` |
| Golden + degradation | `test_golden_verdicts.py`, `test_degradation_matrix.py` |
| Integrations (unit) | `test_integrations_unit.py` |
| ChorusGraph live matrix | `test_chorusgraph_live_matrix.py` (needs `chorusgraph`) |
| P0–P3 / RuntimeAdapter | `test_p0_p3.py`, `test_runtime_conformance.py` |
| BYO runtime (no ChorusGraph) | `test_byo_runtime.py` |
| README examples | `test_readme_examples.py` |
| CLI / cache / calibrate | `test_cache_audit_cli.py`, `test_calibrate_spans_judge.py` |
| Benchmarks | `benchmarks/` |

```bash
pip install -e ".[dev]"
pytest --cov=prismshine
```

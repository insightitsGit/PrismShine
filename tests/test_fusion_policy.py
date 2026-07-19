from __future__ import annotations

from prismshine.fusion import fuse
from prismshine.models import Signal, SignatureHit
from prismshine.policy import bump_strictness, resolve_policy


def test_strictness_band_shift():
    std = resolve_policy(strictness="standard")
    strict = resolve_policy(strictness="strict")
    assert strict.bands[0] < std.bands[0]


def test_override_precedence():
    pol = resolve_policy(
        profile="default",
        strictness="standard",
        overrides={"tau_sent": 0.99},
    )
    assert pol.tau_sent == 0.99


def test_dynamic_bump():
    assert bump_strictness("standard", 1) == "strict"
    pol = resolve_policy(strictness="standard", dynamic_bump=1)
    assert pol.strictness_effective == "strict"


def test_fusion_pass_and_block():
    pol = resolve_policy()
    low = fuse(
        [Signal(name="grounding.risk_coverage", tier=2, value=0.05, weight=0.25)],
        [],
        pol,
    )
    assert low.decision == "pass"
    fatal = fuse(
        [],
        [SignatureHit(id="EMPTY_RETRIEVAL", severity="fatal", advice="x")],
        pol,
        has_fatal_halt=True,
        early_gate="HANDBOOK:EMPTY_RETRIEVAL",
    )
    assert fatal.resolution_gate == "HANDBOOK:EMPTY_RETRIEVAL"

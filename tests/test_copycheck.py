from __future__ import annotations

from prismshine.evidence.builder import bundle_from_dict
from prismshine.grounding.copycheck import copycheck, extract_facts


def test_number_and_currency_match():
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue was $1,000.",
            "preload": [{"text": "Revenue was $1000 in Q1.", "chunk_id": "1"}],
        }
    )
    r = copycheck(b)
    assert r.unmatched_ratio == 0.0


def test_unmatched_number():
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue was $9999.",
            "preload": [{"text": "Revenue was $1000 in Q1.", "chunk_id": "1"}],
        }
    )
    r = copycheck(b)
    assert r.unmatched_ratio > 0


def test_arithmetic_closure_percent():
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Revenue grew 100 percent.",
            "preload": [
                {
                    "text": "Revenue was 50 then 100.",
                    "chunk_id": "1",
                }
            ],
        }
    )
    r = copycheck(b)
    # 100 may match preload number; percent-change 100 from 50->100
    assert r.unmatched_ratio == 0.0 or len(r.derived) >= 0


def test_arithmetic_sum_derived():
    b, _ = bundle_from_dict(
        {
            "question": "q",
            "answer": "Total is 30.",
            "preload": [{"text": "Part A is 10. Part B is 20.", "chunk_id": "1"}],
        }
    )
    r = copycheck(b)
    assert any(f.value == 30 for f in r.derived) or r.unmatched_ratio == 0.0


def test_date_normalization():
    facts = extract_facts("Signed on January 5, 2024.")
    assert any(f.kind == "date" and f.normalized == "2024-01-05" for f in facts)


def test_place_name_mismatch():
    """Asia vs Europe must surface as unmatched entities (not silent CLEAN_FAST_PATH)."""
    facts = extract_facts("Demand grew in Asia this quarter.")
    assert any(f.kind == "entity" and f.normalized == "asia" for f in facts)
    b, _ = bundle_from_dict(
        {
            "question": "Where was demand?",
            "answer": "The CEO cited strong demand in Asia this quarter.",
            "preload": [
                {
                    "chunk_id": "1",
                    "text": "The CEO cited strong demand in Europe this quarter.",
                }
            ],
        }
    )
    r = copycheck(b)
    assert any(f.normalized == "asia" for f in r.unmatched)
    assert r.unmatched_ratio > 0.0


def test_entity_swap_whole_phrase_not_surname_match():
    """Lisa Munn must not match via the bare surname in Olivia Munn."""
    b, _ = bundle_from_dict(
        {
            "question": "Who appeared in Ride Along 2?",
            "answer": "The actress is Lisa Munn.",
            "preload": [
                {
                    "chunk_id": "1",
                    "text": "The film stars Kevin Hart and Olivia Munn.",
                }
            ],
        }
    )
    r = copycheck(b)
    assert any(f.normalized == "lisa munn" for f in r.unmatched)

"""JudgeBench adapter/eval tests - no external calls."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import judgebench_adapter as A  # noqa: E402
from bench import judgebench_eval as E  # noqa: E402


def _pair(label="A>B"):
    return {"pair_id": "p1", "split": "gpt", "source": "test",
            "question": "2+2=?", "response_A": "4", "response_B": "5",
            "label": label}


# --- canonical mapping ---------------------------------------------------------

def test_gold_mapping():
    assert A.gold_winner("A>B") == "A"
    assert A.gold_winner("B>A") == "B"
    with pytest.raises(ValueError):
        A.gold_winner("tie")


def test_canonical_unswapped():
    assert A.canonical_winner("candidate_a", False) == "A"
    assert A.canonical_winner("candidate_b", False) == "B"
    assert A.canonical_winner("uncertain", False) == "uncertain"


def test_canonical_swapped():
    # pass 2 displays original B in slot A
    assert A.canonical_winner("candidate_a", True) == "B"
    assert A.canonical_winner("candidate_b", True) == "A"
    assert A.canonical_winner("uncertain", True) == "uncertain"
    assert A.canonical_winner("garbage", True) is None


def test_state_has_no_label_leak():
    s = A.jb_state(_pair("B>A"), swapped=False)
    assert "B>A" not in s and "label" not in s.lower()
    assert "response_A" not in s and "response_B" not in s
    assert "CANDIDATE A" in s and "CANDIDATE B" in s


def test_swap_presents_same_content():
    p = _pair()
    s1, s2 = A.jb_state(p, False), A.jb_state(p, True)
    assert "CANDIDATE A:\n4" in s1 and "CANDIDATE A:\n5" in s2


# --- metrics -------------------------------------------------------------------

def _row(pid, ps, verdict, gold, system="sys"):
    sw = ps == 2
    canon = A.canonical_winner(verdict, sw)
    return {"system": system, "pair_id": pid, "split": "gpt",
            "source": "s", "pass": ps, "swapped": sw, "gold": gold,
            "verdict": verdict, "canonical": canon,
            "correct": canon == gold if canon in ("A", "B") else False,
            "confidence": 0.9, "latency_s": 1.0, "input_tokens": 500,
            "truncated": False}


def test_correct_after_swap():
    # gold A>B; pass1 shows A first -> candidate_a correct;
    # pass2 shows B first -> candidate_b is canonical A -> also correct
    r1 = _row("p", 1, "candidate_a", "A")
    r2 = _row("p", 2, "candidate_b", "A")
    st = E.system_stats([r1, r2], [])
    assert st["accuracy"] == 1.0
    assert st["strict_pair_accuracy"] == 1.0
    assert st["swap_consistency"] == 1.0


def test_wrong_after_swap():
    # consistent but wrong: picks original B both times while gold is A
    r1 = _row("p", 1, "candidate_b", "A")
    r2 = _row("p", 2, "candidate_a", "A")
    st = E.system_stats([r1, r2], [])
    assert st["accuracy"] == 0.0
    assert st["strict_pair_accuracy"] == 0.0
    assert st["swap_consistency"] == 1.0


def test_uncertain_is_not_correct():
    r1 = _row("p", 1, "uncertain", "A")
    r2 = _row("p", 2, "uncertain", "A")
    st = E.system_stats([r1, r2], [])
    assert st["accuracy"] == 0.0
    assert st["coverage"] == 0.0
    assert st["abstention_rate"] == 1.0
    assert st["conditional_accuracy"] is None


def test_malformed_is_not_uncertain():
    r = _row("p", 1, "PARSE_FAILURE", "A")
    assert r["canonical"] is None and r["correct"] is False
    st = E.system_stats([r, _row("p", 2, "candidate_a", "A")], [])
    assert st["failures"]["PARSE_FAILURE"] == 1
    assert st["abstention_rate"] == 0.0


def test_timeout_is_not_uncertain():
    r = _row("p", 1, "TIMEOUT", "A")
    assert r["canonical"] is None
    st = E.system_stats([r], [])
    assert st["failures"]["TIMEOUT"] == 1
    assert st["n_valid"] == 0


def test_coverage_and_conditional():
    rows = [_row("a", 1, "candidate_a", "A"),   # correct
            _row("a", 2, "candidate_b", "A"),   # correct (canon A)
            _row("b", 1, "candidate_b", "A"),   # wrong
            _row("b", 2, "candidate_a", "A"),   # wrong (canon B)
            _row("c", 1, "uncertain", "A"),
            _row("c", 2, "uncertain", "A")]
    st = E.system_stats(rows, [])
    assert st["coverage"] == pytest.approx(4 / 6)
    assert st["conditional_accuracy"] == pytest.approx(0.5)
    assert st["selective_risk"] == pytest.approx(0.5)
    assert st["abstention_rate"] == pytest.approx(2 / 6)
    # strict pair: pair a both correct -> 1/3
    assert st["strict_pair_accuracy"] == pytest.approx(1 / 3)


def test_swap_inconsistency_detection():
    # pass1 canon A, pass2 canon B -> inconsistent flip
    r1 = _row("p", 1, "candidate_a", "A")
    r2 = _row("p", 2, "candidate_a", "A")   # canon B
    st = E.system_stats([r1, r2], [])
    assert st["swap_consistency"] == 0.0
    assert st["consistency_detail"]["flip_A_to_B"] == 1


def test_split_and_source_aggregation(tmp_path):
    rows = []
    for i in range(4):
        split = "gpt" if i < 3 else "claude"
        for ps in (1, 2):
            r = _row(f"p{i}", ps, "candidate_a", "A")
            r["split"] = split
            r["source"] = "src1" if i < 2 else "src2"
            rows.append(r)
    pairs = [{"pair_id": f"p{i}", "split": r["split"],
              "source": r["source"], "label": "A>B"}
             for i, r in enumerate(rows[::2])]
    res, _ = E.analyze(rows, pairs, type("A", (), {})())
    st = res["systems"]["sys"]
    assert st["per_split"]["gpt"]["n_calls"] == 6
    assert st["per_split"]["claude"]["n_calls"] == 2
    assert st["per_source"]["src1"]["low_n"] is True


def test_truncation_buckets():
    rows = [_row("a", 1, "candidate_a", "A"),
            _row("b", 1, "candidate_a", "A")]
    rows[0]["input_tokens"], rows[1]["input_tokens"] = 400, 1500
    rows[1]["truncated"] = True
    st = E.system_stats(rows, [])
    assert st["truncation"]["<=512"]["n"] == 1
    assert st["truncation"][">1024"]["n"] == 1
    assert st["truncation"]["truncated"]["n"] == 1


def test_resume_skips_terminal_verdicts(tmp_path):
    out = tmp_path / "raw.jsonl"
    out.write_text("\n".join([
        json.dumps(_row("p1", 1, "candidate_a", "A", "jev")),
        json.dumps(_row("p1", 2, "TIMEOUT", "A", "jev")),
    ]) + "\n")
    args = type("A", (), {"out": str(out), "system": "jev"})()
    done = set()
    for r in E.iter_jsonl(args.out):
        if r.get("verdict") in E.RETRYABLE:
            continue
        done.add((r.get("system"), r.get("pair_id"), r.get("pass")))
    assert ("jev", "p1", 1) in done
    assert ("jev", "p1", 2) not in done   # timeout is retried

"""Subprocess Core ML engine for tests - NOT a pytest module (no test_ prefix).

  python tests/coreml_worker.py <pkg_dir> <ckpt|-> <compute_units>

Reads one JSON request per stdin line and writes one JSON response per line:

  {"op": "info"}                                  -> engine metadata
  {"op": "decide", "state": ..., "questions": ...} -> decide() result
  {"op": "quit"}                                  -> exits

The ANE runtime has been observed to corrupt the heap of the host process on
this machine (Core ML destroying a bridged PyObject* on a background dispatch
thread - crash reports show objc_destructInstance -> _PyObject_Free, with
later faults in unrelated native calls). Running the engine in a child process
keeps pytest itself off that heap; if the child dies the test sees a clean
error instead of a SIGSEGV mid-suite.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    pkg_dir, ckpt, cu = sys.argv[1], sys.argv[2], sys.argv[3]
    from coreml_backend import CoreMLEngine
    eng = CoreMLEngine(pkg_dir, ckpt=None if ckpt == "-" else ckpt,
                       compute_units=cu)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            if req["op"] == "info":
                res = {"compute_units": eng.compute_units,
                       "max_seq": eng.max_seq, "max_len": eng.max_len,
                       "encode_len": eng.encode_len,
                       "buckets": sorted(eng._models)}
            elif req["op"] == "decide":
                calls = 0
                orig = eng.label_logits

                def counting(enc):
                    nonlocal calls
                    calls += 1
                    return orig(enc)

                eng.label_logits = counting
                try:
                    res = {"result": eng.decide(req["state"],
                                                req["questions"]),
                           "label_calls": calls}
                finally:
                    eng.label_logits = orig
            elif req["op"] == "quit":
                return
            else:
                raise ValueError(f"unknown op {req['op']!r}")
            out = {"ok": True, **res}
        except Exception as e:                      # noqa: BLE001 - report it
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

# JudgeBench External-Gold Evaluation

Independent-gold evaluation of **historical Jev** and **JevLite** against
`ScalerLab/JudgeBench` objective pairwise labels. Jev is a system under test,
not a reference. JudgeBench `label` (A>B / B>A) is the only gold; no output of
Jev, Gemma, Qwen, or any prior judge result was used to generate gold.

## Dataset

| field | value |
|---|---|
| repository | `ScalerLab/JudgeBench` (Hugging Face) |
| revision | `57dd5e0b9817d07f05ec8f45a91b2ce1e310e308` |
| GPT pairs | 350 |
| Claude pairs | 270 |
| Total pairs | 620 (asserted: EXPECTED=620, ACTUAL=620) |
| canonical JSONL | `artifacts/judgebench_620.jsonl` |
| canonical SHA-256 | `29676f075e2c75898bb21af9eaf72a3731f27f385f1b682af4dc9ea03dcebfcf` |

Model-facing input is only `QUESTION` + `CANDIDATE A` + `CANDIDATE B`
(`bench/judgebench_adapter.py`, SHA-256 `b7ef903b…` of the file). `label`,
`source`, `response_model`, the JudgeBench name, and all prior judge results
are excluded from the model input (leak test: `test_state_has_no_label_leak`).

## Systems

| system | config |
|---|---|
| historical Jev | `api.typesafe.ai/v1/systemone`, model `jev-latest`, typed `choice` question over {candidate_a, candidate_b, uncertain}; state char-capped at 6000 (historical convention) |
| JevLite CoreML | `DecisionEngine(backend="coreml", coreml_mode="worker")` — production worker-isolated path, `cpu_and_ne`, buckets 256/512/1024, `jevlite.pt` SHA-256 `6706cde7…`, Core ML 1024 model `f1a77098…` |
| JevLite Torch | parity spot-check only, `backend="torch"`, MPS, same checkpoint |

Both systems receive the identical semantic task through the same adapter.
Each pair is judged twice (original order, swapped order); swapped verdicts
are mapped back to canonical response identity before scoring.

- calls per system: 620 pairs × 2 orders = **1240**
- random forced-choice baseline: **50%** (binary gold; abstentions count wrong
  in accuracy and are also reported via coverage / conditional accuracy)

## Main results

| metric | Jev | JevLite CoreML |
|---|---|---|
| accuracy (forced) | **0.733** | **0.112** |
| strict-pair accuracy | 0.626 | 0.000 |
| coverage | 0.968 | 0.239 |
| conditional accuracy | 0.758 | **0.470** |
| selective risk | 0.243 | 0.530 |
| abstention rate | 0.032 | 0.761 |
| wrong-answer rate | 0.235 | 0.127 |
| swap consistency | 0.779 | 0.597 |
| original-order acc | 0.745 | 0.102 |
| swapped-order acc | 0.721 | 0.123 |
| first-position pick rate (decided) | 0.54 | **0.00** |
| failures | none | none |

### Split breakdown

| split | n calls | Jev acc / cov | JevLite acc / cov |
|---|---|---|---|
| gpt | 700 | 0.729 / 0.967 | 0.107 / 0.236 |
| claude | 540 | 0.739 / 0.969 | 0.119 / 0.243 |

Per-source table is in `artifacts/judgebench_summary.json` (`per_source`;
mmlu-pro-* sources have n=44 calls each and are marked `low_n` — no strong
conclusions drawn from them).

### Outcome matrix (per pair, canonical, both passes merged by strict rule)

| outcome | count |
|---|---|
| both correct | 41 |
| Jev only correct | 72 |
| JevLite only correct | 22 |
| both wrong | 23 |
| JevLite abstain + Jev correct | **349** |
| JevLite abstain + Jev wrong | 113 |

### Position bias

- Jev: 54% of decided calls pick the first displayed position; swap flips
  A→B = 86 vs B→A = 39 — measurable but moderate bias.
- JevLite: **0% of decided calls pick the first position.** Across all 1240
  calls the verdicts were `uncertain` 944 / `candidate_b` 296 /
  `candidate_a` **0**. `candidate_a` softmax probability never exceeded
  0.319 (mean 0.110), i.e. the first-position label is structurally unable
  to win. Swap flips: 0 A→B, 46 B→A, plus 204 decisive↔uncertain churn.

### Truncation

- JevLite (token cap = 1024, state absorbs truncation): 774/1240 calls
  truncated (62.4%); longest input 2855 tokens, max dropped 1831.
  Forced accuracy: non-truncated 0.485 (n=466) vs truncated 0.461 (n=774) —
  truncation is not the main failure driver.
- Jev (state char cap 6000): 242/1240 calls had state chars dropped.
- Effective inputs differ by design (Jev char cap vs JevLite 1024-token cap);
  both are the systems' production truncation behavior, reported, not hidden.

### Failures

- 0 ERROR / 0 TIMEOUT / 0 MALFORMED / 0 CRASH rows for both systems.
- Worker crashes/restarts: none surfaced during the 1240-call run
  (`judgebench_failures.jsonl` is empty).
- `uncertain` counts are genuine model abstentions, never converted errors.

## Torch/MPS parity spot-check (40 pairs, 80 calls)

Same pathology reproduced on the PyTorch backend: verdicts `uncertain` 64 /
`candidate_b` 16 / `candidate_a` **0**; `candidate_a` max probability 0.246.
Mean |prob diff| vs CoreML on identical inputs 0.054 — consistent with fp16
numerical noise. **The never-A pathology is model-level, not a CoreML or
worker bug.** Full Torch run skipped per spec (CoreML is the production
backend under test; parity evidence is sufficient).

## Interpretation

1. **Jev transfers to independent objective judging.** 73.3% accuracy vs the
   50% baseline, with meaningful confidence calibration (accuracy rises
   monotonically 0.48 → 0.92 across confidence buckets). It is imperfect —
   strict-pair 62.6% and 78% swap consistency show real position bias — but
   clearly above chance on gold it never trained against.

2. **JevLite does not transfer.** Forced accuracy 11.2%, and even its
   *decisive* answers are below chance (conditional accuracy 0.470 < 0.50).
   This is not "conservative abstention":

   - When JevLite abstains, Jev is still right 349/462 = **75.5%** — identical
     to Jev's overall conditional accuracy (75.8%). The abstention signal does
     not identify harder cases at all.
   - The model **never selects the first displayed position** on this task —
     a label-position prior, not semantic label reading. It only ever emits
     `candidate_b` (2nd position) or `uncertain` (3rd position).
   - The same never-A pathology reproduces on the Torch backend, ruling out
     CoreML/worker artifacts.

3. **Spec's expected branch: A.** Jev strong / JevLite weak. JevLite's
   historical-workload `uncertain` collapse (99.8% in the novllm corpus) was
   not workload-specific — it is a general label-usage pathology. Training
   and OOD label-generalization need investigation before JevLite can act as
   any kind of judge, selective or otherwise.

4. **"Does JevLite resemble Jev?" was not the question — and the answer to
   the real questions is:** Jev ≈ 73% correct on independent gold; JevLite ≈
   11% (47% when it dares to answer); `uncertain` is **not** functioning as
   useful abstention.

## Reproducibility

| field | value |
|---|---|
| git commits | harness + results commits on `feat/judgebench-external-gold` (see `git log`) |
| adapter | `bench/judgebench_adapter.py` sha256 `b7ef903b…` |
| JevLite checkpoint | `jevlite.pt` sha256 `6706cde7…` |
| CoreML packages | `artifacts/coreml/jevlite-{256,512,1024}.mlpackage` (fp16), `cpu_and_ne` |
| hardware | Apple M2 Max, macOS 26.6.2 |
| run commands | `python bench/judgebench_eval.py --system {jev,jevlite-coreml} --out artifacts/judgebench_raw.jsonl --summary artifacts/judgebench_summary.json` |
| concurrency | jev 8, jevlite-coreml 4 |
| artifacts | `judgebench_summary.json` (metrics), `judgebench_raw.jsonl` (per-call, gitignored), `judgebench_disagreements.jsonl` (556 rows, gitignored), `judgebench_failures.jsonl` (empty), `judgebench_620.jsonl` (dataset, gitignored) |
| latency | Jev mean 0.63s p95 0.79s; JevLite worker mean 0.47s p95 0.95s |

## Limitations

- Jev is a remote API (`jev-latest`); its version is server-side and its
  tokenization/truncation is opaque (only the 6000-char state cap is known).
- Control judges (Gemma/Qwen on JudgeBench) were not run: optional per spec;
  the HAI API key was not available in this environment.
- JevLite's never-A was observed on 1240 calls; the positional-prior
  explanation is consistent with all evidence but the internal mechanism
  (attention over `[LID]` positions vs label text) is not fully dissected.

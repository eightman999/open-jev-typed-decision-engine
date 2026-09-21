"""Base-model vs checkpoint positional-prior probe.

Is the first-label-position suppression (never-A) a failure of the current
jevlite.pt fine-tune, or a property of the forked encoder
(answerdotai/ModernBERT-base) combined with this serialization?

  A  base encoder + frozen weights + fresh Linear(d,1) head, trained on
     synthetic 2..6-way choice with uniform gold position and randomized
     label names (never JudgeBench data)
  B  existing jevlite.pt, evaluation only

Synthetic task (both train and eval): the state shows K CANDIDATE blocks for a
known-answer fact question; exactly one candidate is correct. Each label's
description names the candidate letter it stands for; label names are random.
Position alone carries no signal.

    QUESTION: What is 2+2?
    CANDIDATE A: 2+2 equals 9.
    CANDIDATE B: 2+2 equals 4.
    CANDIDATE C: 2+2 equals 12.

    criteria (sorted): lbl_aa -> "candidate A is objectively more correct"
                       lbl_fo -> "candidate B is objectively more correct" # gold
                       lbl_zz -> "candidate C is objectively more correct"

Eval additionally emits a rotated arrangement per case (descriptions rotated
one position, same keys) -> permutation consistency = same key chosen.

Usage:
  python bench/posprior_probe.py --make-data          # train + eval sets
  python bench/posprior_probe.py --train-a            # condition A
  python bench/posprior_probe.py --eval jevlite.pt    # condition B
  python bench/posprior_probe.py --eval artifacts/posprior_a.pt
"""
from __future__ import annotations
import argparse
import json
import os
import random
import string
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ART = "artifacts"
TRAIN_F = f"{ART}/posprior_train.jsonl"
EVAL_F = f"{ART}/posprior_eval.jsonl"
METRICS_F = f"{ART}/posprior_metrics.json"
CKPT_A = f"{ART}/posprior_a.pt"
SEED = 20260922
SIZES = (2, 3, 4, 5, 6)
N_TRAIN = 800
N_EVAL_PER_SIZE = 120
MAX_LEN = 256


def _rid(rng, n=5):
    return "".join(rng.choice(string.ascii_lowercase + string.digits)
                   for _ in range(n))


_WORDS = (
    "apple amber anchor arrow autumn bamboo beacon birch breeze brook canyon "
    "cedar cherry cliff cloud clover comet coral creek crystal dawn delta "
    "ember falcon fern flint forest frost garnet glacier harbor hazel holly "
    "island ivory jasper juniper lagoon laurel lily linden lotus maple marble "
    "meadow mesa mist moss north oak ocean olive onyx opal orchid otter pearl "
    "pebble pine plum prism quartz raven reed river robin sage sand sapphire "
    "shadow silver slate solar sparrow spring stone storm summit timber topaz "
    "tulip twilight valley velvet violet walnut willow winter wren zephyr"
).split()

_FACTS = [
    ("What is 2+2?", "2+2 equals 4.", "2+2 equals {w}."),
    ("What is the capital of France?", "The capital of France is Paris.",
     "The capital of France is {w}."),
    ("How many days are in a week?", "There are 7 days in a week.",
     "There are {w} days in a week."),
    ("What color is the sky on a clear day?", "The sky is blue.",
     "The sky is {w}."),
    ("Is water made of hydrogen and oxygen?", "Yes, water is H2O.",
     "No, water is {w}."),
    ("What is 10 - 4?", "10 minus 4 equals 6.", "10 minus 4 equals {w}."),
    ("What is the largest planet?", "Jupiter is the largest planet.",
     "{w} is the largest planet."),
    ("What is the chemical symbol for gold?", "The symbol for gold is Au.",
     "The symbol for gold is {w}."),
]
_WRONG = ["5", "9", "Madrid", "Rome", "green", "orange", "CO2", "HCl",
          "Venus", "Mars", "Ag", "Pb", "12", "0", "Wednesday", "April"]

_LETTERS = "ABCDEF"


def make_case(rng, k, cid):
    """JudgeBench-shaped synthetic case: K candidates, exactly one correct.

    state shows CANDIDATE A..; each label's description names the candidate
    letter it stands for; label names are random. gold = the label whose
    description mentions the letter of the correct candidate.
    """
    q, good, bad = rng.choice(_FACTS)
    texts = [bad.format(w=w) for w in rng.sample(_WRONG, k)]
    gold_pos = rng.randrange(k)
    texts[gold_pos] = good
    names = sorted(f"lbl_{_rid(rng)}" for _ in range(k))
    state = ("QUESTION:\n" + q + "\n\n" + "\n\n".join(
        f"CANDIDATE {_LETTERS[i]}:\n{texts[i]}" for i in range(k)))
    criteria = {names[i]: f"candidate {_LETTERS[i]} is objectively "
                         f"more correct" for i in range(k)}
    return {"id": cid, "state": state,
            "questions": {"judge": {
                "type": "choice",
                "instructions": "Choose the label of the candidate that is "
                                "objectively more correct.",
                "criteria": criteria}},
            "gold": {"judge": {"label": names[gold_pos],
                               "probabilities": {
                                   n: 1.0 if i == gold_pos else 0.0
                                   for i, n in enumerate(names)}}},
            "k": k, "gold_pos": gold_pos, "gold_key": _LETTERS[gold_pos],
            "pos_key": {str(i): _LETTERS[i] for i in range(k)}}


def rotated(case):
    """Same case with descriptions rotated one position: the gold meaning
    (key) moves to a different sorted position."""
    k = case["k"]
    crit = case["questions"]["judge"]["criteria"]
    names = sorted(crit)
    descs = [crit[n] for n in names]
    rot = descs[1:] + descs[:1]
    new_crit = {names[i]: rot[i] for i in range(k)}
    # gold key now sits where its description moved to
    old_gold = names.index(case["gold"]["judge"]["label"])
    new_pos = (old_gold - 1) % k
    q = {"judge": {"type": "choice",
                   "instructions": case["questions"]["judge"]["instructions"],
                   "criteria": new_crit}}
    return {"id": case["id"] + "_r", "state": case["state"],
            "questions": q,
            "gold": {"judge": {"label": names[new_pos],
                               "probabilities": {
                                   n: 1.0 if i == new_pos else 0.0
                                   for i, n in enumerate(names)}}},
            "k": k, "gold_pos": new_pos, "gold_key": case["gold_key"],
            "pos_key": {str(i): case["pos_key"][str((i + 1) % k)]
                        for i in range(k)},
            "rot_of": case["id"]}


def make_data():
    rng = random.Random(SEED)
    train, ev = [], []
    for i in range(N_TRAIN):
        train.append(make_case(rng, rng.choice(SIZES), f"tr{i:04d}"))
    for k in SIZES:
        for i in range(N_EVAL_PER_SIZE):
            c = make_case(rng, k, f"k{k}_e{i:04d}")
            ev.append(c)
            ev.append(rotated(c))
    with open(TRAIN_F, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(EVAL_F, "w") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dist = Counter(r["gold_pos"] for r in train)
    by_k = Counter((r["k"], r["gold_pos"]) for r in train)
    print(f"train {len(train)} -> {TRAIN_F} | gold_pos dist {dict(sorted(dist.items()))}")
    print(f"per-size gold_pos: {dict(sorted(by_k.items()))}")
    print(f"eval {len(ev)} -> {EVAL_F} (incl. rotated variants)")


# --- model plumbing --------------------------------------------------------------

def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _marker_rows_only(model):
    """Freeze every encoder weight; leave only the two new marker-token
    embedding rows trainable (they were never pretrained)."""
    import torch
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    emb = model.encoder.get_input_embeddings().weight
    emb.requires_grad_(True)
    mask = torch.zeros_like(emb)
    mask[[model.qid, model.lid]] = 1.0
    emb.register_hook(lambda g: g * mask)


def train_a(epochs=3, bs=8, accum=1, lr=1e-3, enc_lr=3e-5, freeze=True):
    import torch
    from torch.utils.data import DataLoader
    from td_data import TypedDecisions, collate
    from device import pick_device
    from model import JevLite, decision_loss, DEFAULT_ENCODER

    torch.manual_seed(SEED)
    dev = pick_device()
    model = JevLite(DEFAULT_ENCODER).to(dev)
    if freeze:
        _marker_rows_only(model)
    groups = ([{"params": [p for n, p in model.named_parameters()
                           if n.startswith("score.")], "lr": lr},
               {"params": [p for n, p in model.named_parameters()
                           if not n.startswith("score.") and p.requires_grad],
                "lr": enc_lr}]
              if not freeze else
              [{"params": [p for p in model.parameters() if p.requires_grad],
                "lr": lr}])
    rows = load_rows(TRAIN_F)
    ds = TypedDecisions(rows, model.tok, MAX_LEN, model.qid, model.lid)
    dl = DataLoader(ds, batch_size=bs, shuffle=True,
                    collate_fn=lambda b: collate(b, model.tok.pad_token_id))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    print(f"A({'frozen' if freeze else 'full'}): device={dev} "
          f"trainable_params={sum(p.numel() for p in params)} "
          f"| rows={len(rows)} | eff. batch {bs * accum}")
    for ep in range(epochs):
        model.train()
        run, t0 = 0.0, time.time()
        for i, b in enumerate(dl):
            b = {k: v.to(dev) for k, v in b.items()}
            _, probs = model(b)
            loss, _, _ = decision_loss(probs, b, 1.0)
            (loss / accum).backward()
            run += loss.item()
            if (i + 1) % accum == 0 or i + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        acc = probe_accuracy(model, dev)
        print(f"  epoch {ep + 1}/{epochs} loss {run / len(dl):.4f} "
              f"eval-subset acc {acc:.3f} {time.time() - t0:.0f}s",
              flush=True)
    out = CKPT_A if freeze else CKPT_A.replace(".pt", "_full.pt")
    torch.save({"state_dict": model.state_dict(), "encoder": DEFAULT_ENCODER,
                "max_len": MAX_LEN, "temperatures": [1.0, 1.0, 1.0],
                "probe": f"{'frozen' if freeze else 'full'}-encoder "
                         "synthetic choice"}, out)
    print(f"wrote {out}")


def probe_accuracy(model, dev, n=100):
    """Quick accuracy check on a fixed eval subset sampled across sizes."""
    import torch
    import td_data
    rows = load_rows(EVAL_F)
    rows = [r for i, r in enumerate(rows) if i % (len(rows) // n) == 0][:n]
    model.eval()
    hit = 0
    for r in rows:
        enc = td_data.encode(r, model.tok, MAX_LEN, model.qid, model.lid,
                             with_gold=False)
        b = td_data.collate([enc], model.tok.pad_token_id)
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.no_grad():
            _, probs = model(b)
        labels = enc["labels"][0]
        if labels[probs[0][:len(labels)].argmax().item()] == \
                r["gold"]["judge"]["label"]:
            hit += 1
    model.train()
    return hit / len(rows)


# --- evaluation -------------------------------------------------------------------

def _load_any(path, dev):
    import torch
    from model import JevLite
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = JevLite(ck["encoder"])
    m.load_state_dict(ck["state_dict"])
    m.to(dev).eval()
    return m, ck


def eval_model(path):
    """Per-size position stats + permutation consistency for one checkpoint.

    path == "UNTRAINED" evaluates a freshly initialized JevLite
    (base encoder + random head, no training at all)."""
    import torch
    import td_data
    from device import pick_device

    dev = pick_device()
    if path == "UNTRAINED":
        from model import JevLite, DEFAULT_ENCODER
        torch.manual_seed(SEED)
        m = JevLite(DEFAULT_ENCODER).to(dev).eval()
    else:
        m, _ = _load_any(path, dev)
    rows = load_rows(EVAL_F)
    per_size = defaultdict(lambda: {
        "n": 0, "correct": 0, "wins": Counter(),
        "prob_by_pos": defaultdict(list), "logit_by_pos": defaultdict(list),
        "gold_pos_hits": Counter(), "gold_pos_n": Counter(),
        "pos0_selected": 0})
    base_pick = {}   # case_id -> picked key (unrotated arrangement)
    rot_pick = {}    # case_id -> picked key (rotated arrangement)

    for r in rows:
        enc = td_data.encode(r, m.tok, MAX_LEN, m.qid, m.lid,
                             with_gold=False)
        b = td_data.collate([enc], m.tok.pad_token_id)
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.no_grad():
            lg, probs = m(b)
        labels = enc["labels"][0]
        ps = probs[0].tolist()[:len(labels)]
        lg = lg[0].tolist()[:len(labels)]
        pick_i = max(range(len(labels)), key=lambda i: ps[i])
        gold_label = r["gold"]["judge"]["label"]
        gold_i = labels.index(gold_label)

        s = per_size[r["k"]]
        s["n"] += 1
        s["correct"] += int(pick_i == gold_i)
        s["wins"][pick_i] += 1
        s["pos0_selected"] += int(pick_i == 0)
        s["gold_pos_n"][gold_i] += 1
        s["gold_pos_hits"][gold_i] += int(pick_i == gold_i)
        for i in range(len(labels)):
            s["prob_by_pos"][i].append(ps[i])
            s["logit_by_pos"][i].append(lg[i])

        key = r["pos_key"][str(pick_i)]
        (rot_pick if r.get("rot_of") else base_pick)[
            r.get("rot_of") or r["id"]] = key

    out = {"checkpoint": path, "per_size": {}}
    for k, s in sorted(per_size.items()):
        out["per_size"][str(k)] = {
            "n": s["n"],
            "accuracy": s["correct"] / s["n"],
            "win_rate_by_pos": {str(i): s["wins"][i] / s["n"]
                                for i in range(k)},
            "pos0_selection_rate": s["pos0_selected"] / s["n"],
            "mean_prob_by_pos": {str(i): sum(v) / len(v)
                                 for i, v in
                                 sorted(s["prob_by_pos"].items())},
            "mean_logit_by_pos": {str(i): sum(v) / len(v)
                                  for i, v in
                                  sorted(s["logit_by_pos"].items())},
            "acc_by_gold_pos": {str(i): s["gold_pos_hits"][i]
                                / s["gold_pos_n"][i]
                                for i in sorted(s["gold_pos_n"])},
        }
    same = sum(1 for cid, key in base_pick.items()
               if rot_pick.get(cid) == key)
    out["permutation_consistency"] = same / max(len(base_pick), 1)
    out["permutation_pairs"] = len(base_pick)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--make-data", action="store_true")
    ap.add_argument("--train-a", action="store_true")
    ap.add_argument("--unfreeze", action="store_true",
                    help="fallback: train the full encoder too (enc 3e-5)")
    ap.add_argument("--enc-lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval", default=None, help="checkpoint path")
    a = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    if a.make_data:
        make_data()
    if a.train_a:
        train_a(epochs=a.epochs, freeze=not a.unfreeze,
                enc_lr=a.enc_lr, lr=a.head_lr, bs=a.bs, accum=a.accum)
    if a.eval:
        res = eval_model(a.eval)
        prev = {}
        if os.path.exists(METRICS_F):
            prev = json.load(open(METRICS_F))
        prev[os.path.basename(a.eval)] = res
        prev["meta"] = {"seed": SEED, "n_train": N_TRAIN,
                        "n_eval_per_size": N_EVAL_PER_SIZE,
                        "sizes": list(SIZES), "max_len": MAX_LEN,
                        "train_file": TRAIN_F, "eval_file": EVAL_F,
                        "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(prev, open(METRICS_F, "w"), indent=1, ensure_ascii=False)
        print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    main()

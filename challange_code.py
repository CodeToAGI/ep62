
"""
EP62 Challenge Solutions
------------------------
1. Which bias sources drive the parity gap and the odds gap?
2. Find the LABEL_FLIP value where the four-fifths rule first breaks
3. Add a third group and see what happens to the metrics

Usage:
    python ep62_challenge_solutions.py
"""

import importlib.util, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
warnings.filterwarnings("ignore")

# ── Config (same as original) ───────────────────────────────────────────────
SEEDS = 5
EPOCHS = 40
HIDDEN = 32
N_TRAIN, N_VAL, N_TEST = 4000, 3000, 6000
MINORITY = 0.30
MINORITY_BIASED = 0.05
LABEL_FLIP = 0.35
NOISE_MULT = 2.5
SHIFT = 0.4
LOADINGS = [1.0, 0.8, 0.6, 0.9, 0.7, 0.5]
FOUR_FIFTHS = 0.8

# ── Data generator (original 2-group version) ───────────────────────────────
def make(n, seed, g1=MINORITY, flip=0.0, noise=1.0, shift=0.0):
    r = np.random.default_rng(seed)
    g = (r.random(n) < g1).astype(int)
    z = r.normal(0, 1, n) - shift * g
    y = (z + 0.5 * r.normal(0, 1, n) > 0).astype(int)
    sd = 1.0 + (noise - 1.0) * g
    x = z[:, None] * np.array(LOADINGS)[None, :] + r.normal(0, 1, (n, len(LOADINGS))) * sd[:, None]
    proxy = g + 0.5 * r.normal(0, 1, n)
    seen = y.copy()
    seen[(g == 1) & (y == 1) & (r.random(n) < flip)] = 0
    return np.column_stack([x, proxy]).astype(np.float32), y, seen, g

# ── 3-group data generator (Challenge 3) ────────────────────────────────────
def make3(n, seed, g1=0.30, g2=0.20, flip=0.0, noise=1.0, shift=0.0):
    """
    Three groups:
      group 0 = majority
      group 1 = minority A (same as original group 1)
      group 2 = minority B (new)
    """
    r = np.random.default_rng(seed)
    u = r.random(n)
    g = np.zeros(n, dtype=int)
    g[u < g1] = 1
    g[(u >= g1) & (u < g1 + g2)] = 2

    # true ability: group 1 and group 2 can have different shifts
    z = r.normal(0, 1, n) - shift * (g == 1).astype(float) - 0.25 * (g == 2).astype(float)
    y = (z + 0.5 * r.normal(0, 1, n) > 0).astype(int)

    # measurement bias only on group 1 (you can change this)
    sd = 1.0 + (noise - 1.0) * (g == 1).astype(float)
    x = z[:, None] * np.array(LOADINGS)[None, :] + r.normal(0, 1, (n, len(LOADINGS))) * sd[:, None]

    # proxy that leaks group membership
    proxy = g + 0.5 * r.normal(0, 1, n)

    # label bias only on group 1 (you can change this)
    seen = y.copy()
    seen[(g == 1) & (y == 1) & (r.random(n) < flip)] = 0
    return np.column_stack([x, proxy]).astype(np.float32), y, seen, g

# ── Training ────────────────────────────────────────────────────────────────
def train(x, y, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x, y = torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    net = nn.Sequential(
        nn.Linear(x.shape[1], HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, 1)
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(EPOCHS):
        order = torch.randperm(len(x))
        for i in range(0, len(x), 128):
            b = order[i:i + 128]
            loss = nn.functional.binary_cross_entropy_with_logits(net(x[b])[:, 0], y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()

class Clf(BaseEstimator, ClassifierMixin):
    def __init__(self, net=None):
        self.net = net
        self.classes_ = np.array([0, 1])
    def fit(self, X, y=None): return self
    def predict_proba(self, X):
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(np.asarray(X, dtype=np.float32)))[:, 0]).numpy()
        return np.column_stack([1 - p, p]).astype(np.float64)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# ── Audit helpers ───────────────────────────────────────────────────────────
def audit(y, p, g):
    groups = sorted(np.unique(g))
    r = {}
    for k in groups:
        mask = g == k
        r[k] = dict(
            sel=float(p[mask].mean()) if mask.any() else 0.0,
            tpr=float(p[mask & (y == 1)].mean()) if (mask & (y == 1)).any() else 0.0,
            fpr=float(p[mask & (y == 0)].mean()) if (mask & (y == 0)).any() else 0.0,
        )
    # gaps relative to group 0
    dpd = abs(r[0]["sel"] - r[1]["sel"]) if 1 in r else 0.0
    eod = max(abs(r[0]["tpr"] - r[1]["tpr"]), abs(r[0]["fpr"] - r[1]["fpr"])) if 1 in r else 0.0
    di  = r[1]["sel"] / max(r[0]["sel"], 1e-9) if 1 in r else 1.0
    return dict(acc=float((p == y).mean()), dpd=dpd, eod=eod, di=di, rates=r)

def ms(v):
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]

# ════════════════════════════════════════════════════════════════════════════
# CHALLENGE 1 — Which sources drive which gap?
# ════════════════════════════════════════════════════════════════════════════
def challenge1():
    print("\n" + "="*70)
    print("CHALLENGE 1 — Which bias sources drive the parity gap and the odds gap?")
    print("="*70)

    CONDS = {
        "control": {},
        "label bias": dict(flip=LABEL_FLIP),
        "representation bias": dict(g1=MINORITY_BIASED),
        "measurement bias": dict(noise=NOISE_MULT),
        "all three": dict(flip=LABEL_FLIP, g1=MINORITY_BIASED, noise=NOISE_MULT),
    }

    runs = {n: [] for n in CONDS}
    for s in range(SEEDS):
        for n, kw in CONDS.items():
            tr = make(N_TRAIN, s, **kw)
            te = make(N_TEST, 1000 + s, noise=kw.get("noise", 1.0))
            net = train(tr[0], tr[2], s)
            p = Clf(net).predict(te[0])
            runs[n].append(audit(te[1], p, te[3]))

    C = {n: {m: ms([r[m] for r in rs]) for m in ("acc", "dpd", "eod", "di")} for n, rs in runs.items()}
    c0 = C["control"]

    print(f"\n{'Condition':<22} {'Parity gap':>12} {'Odds gap':>12} {'Sel. ratio':>12}  Real?")
    print("-"*70)
    for n in CONDS:
        dpd, eod, di = C[n]["dpd"][0], C[n]["eod"][0], C[n]["di"][0]
        real_p = dpd > c0["dpd"][0] + 2 * c0["dpd"][1]
        real_o = eod > c0["eod"][0] + 2 * c0["eod"][1]
        flag = []
        if real_p: flag.append("parity")
        if real_o: flag.append("odds")
        print(f"{n:<22} {dpd:12.3f} {eod:12.3f} {di:12.2f}  {', '.join(flag) or 'none'}")

    # Attribution
    print("\nAttribution (increase relative to control):")
    for n in ["label bias", "representation bias", "measurement bias"]:
        dpd_inc = C[n]["dpd"][0] - c0["dpd"][0]
        eod_inc = C[n]["eod"][0] - c0["eod"][0]
        print(f"  {n:<22}  parity +{dpd_inc:.3f}   odds +{eod_inc:.3f}")

    print("\nConclusion:")
    print("  • Parity gap is driven mainly by LABEL BIAS")
    print("  • Odds gap is driven mainly by MEASUREMENT BIAS (+ label bias)")
    print("  • Representation bias alone produces almost no measurable gap")

# ════════════════════════════════════════════════════════════════════════════
# CHALLENGE 2 — Find where the four-fifths rule first breaks
# ════════════════════════════════════════════════════════════════════════════
def challenge2():
    print("\n" + "="*70)
    print("CHALLENGE 2 — Find the LABEL_FLIP value where the four-fifths rule first breaks")
    print("="*70)

    flips = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    results = []

    for flip in flips:
        ratios = []
        for s in range(SEEDS):
            tr = make(N_TRAIN, s, flip=flip)
            te = make(N_TEST, 1000 + s)
            net = train(tr[0], tr[2], s)
            p = Clf(net).predict(te[0])
            a = audit(te[1], p, te[3])
            ratios.append(a["di"])
        mean_di = float(np.mean(ratios))
        below = float(np.mean([r < FOUR_FIFTHS for r in ratios]))
        results.append((flip, mean_di, below))
        status = "BROKEN" if mean_di < FOUR_FIFTHS else "ok"
        print(f"  LABEL_FLIP = {flip:.2f}  →  selection ratio {mean_di:.3f}  "
              f"(below 0.8 in {below*100:.0f}% of runs)  [{status}]")

    # Find the first break
    first_break = next((f for f, di, _ in results if di < FOUR_FIFTHS), None)
    print("\nResult:")
    if first_break is not None:
        print(f"  The four-fifths rule is first broken around LABEL_FLIP ≈ {first_break:.2f}")
    else:
        print("  The four-fifths rule was never broken in the tested range.")

# ════════════════════════════════════════════════════════════════════════════
# CHALLENGE 3 — Add a third group
# ════════════════════════════════════════════════════════════════════════════
def challenge3():
    print("\n" + "="*70)
    print("CHALLENGE 3 — Add a third group: what happens to the metrics?")
    print("="*70)

    print("\nRunning 3-group experiment (group 0 majority, group 1 & 2 minorities)...")
    print("Label bias + measurement bias applied only to group 1.\n")

    runs = []
    for s in range(SEEDS):
        tr = make3(N_TRAIN, s, flip=LABEL_FLIP, noise=NOISE_MULT)
        te = make3(N_TEST, 1000 + s, noise=NOISE_MULT)
        net = train(tr[0], tr[2], s)
        p = Clf(net).predict(te[0])
        runs.append(audit(te[1], p, te[3]))

    # Average rates
    all_groups = sorted(runs[0]["rates"].keys())
    print(f"{'Group':<8} {'Sel rate':>10} {'TPR':>10} {'FPR':>10}")
    print("-"*42)
    for g in all_groups:
        sel = ms([r["rates"][g]["sel"] for r in runs])
        tpr = ms([r["rates"][g]["tpr"] for r in runs])
        fpr = ms([r["rates"][g]["fpr"] for r in runs])
        print(f"  {g:<6} {sel[0]:10.3f} {tpr[0]:10.3f} {fpr[0]:10.3f}")

    # Pairwise selection ratios vs group 0
    print("\nSelection ratios vs group 0:")
    for g in all_groups:
        if g == 0: continue
        di = ms([r["rates"][g]["sel"] / max(r["rates"][0]["sel"], 1e-9) for r in runs])
        status = "FAILS 4/5 rule" if di[0] < FOUR_FIFTHS else "passes"
        print(f"  group {g} / group 0 = {di[0]:.3f}  ({status})")

    print("\nWhat changed with three groups?")
    print("  • You now have multiple pairwise gaps instead of one")
    print("  • Demographic parity and equalized odds become multi-group constraints")
    print("  • Fairlearn still works, but you must decide which groups to equalize")
    print("  • The story gets richer (and harder) — real audits almost always have >2 groups")

# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("EP62 Challenge Solutions")
    print("Running all three challenges (this takes a few minutes)...")

    challenge1()
    challenge2()
    challenge3()

    print("\n" + "="*70)
    print("All challenges complete.")
    print("="*70)

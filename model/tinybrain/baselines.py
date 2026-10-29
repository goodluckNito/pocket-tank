#!/usr/bin/env python3
"""baselines.py - what else could be the tinybrain? Same data and validation
split as train.py (seed 0, 10%), scored on the same goal-head metrics.

  python3 model/tinybrain/baselines.py --data model/out/tb_labels.jsonl

  - prior:    always answer the training set's mean distribution (the floor)
  - lightgbm: gradient-boosted trees on the same 56 features, fit to the SOFT
              labels (each state repeated once per goal, weighted by its prob)
Needs lightgbm (pip install lightgbm); the MLP itself is train.py."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load  # noqa: E402


def score(name, Q, P, extra=""):
    t1 = P.argmax(1)
    top2 = np.argsort(-Q, 1)[:, :2]
    kl = (P * (np.log(P + 1e-9) - np.log(Q + 1e-9))).sum(1).mean()
    tv = 0.5 * np.abs(P - Q).sum(1).mean()
    print(f"{name:28s} top1 {np.mean(Q.argmax(1) == t1):.3f}  in-top2 {np.mean((top2 == t1[:, None]).any(1)):.3f}"
          f"  KL {kl:.4f}  TV {tv:.3f}  {extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+", default=["model/out/tb_labels.jsonl"])
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--leaves", type=int, default=31)
    args = ap.parse_args()
    X, P, _, _ = load(args.data)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); nv = int(len(X) * 0.1)
    va, tr = idx[:nv], idx[nv:]
    score("prior (mean distribution)", np.tile(P[tr].mean(0), (len(va), 1)), P[va])

    import lightgbm as lgb
    keep = P[tr] > 1e-3
    rows, cls = np.nonzero(keep)
    Xr, yr, wr = X[tr][rows], cls, P[tr][rows, cls]
    for rounds, leaves in ((args.rounds // 4, 15), (args.rounds, args.leaves)):
        m = lgb.LGBMClassifier(objective="multiclass", num_class=8, n_estimators=rounds, num_leaves=leaves,
                               learning_rate=0.08, min_child_samples=20, verbose=-1)
        m.fit(Xr, yr, sample_weight=wr)
        Q = np.zeros((len(va), 8)); Q[:, m.classes_] = m.predict_proba(X[va])
        n_nodes = m.booster_.trees_to_dataframe().shape[0]
        score(f"lightgbm {rounds}x{leaves} leaves", Q, P[va],
              f"({rounds * len(m.classes_)} trees, {n_nodes} nodes ~ {n_nodes * 8 // 1024} KB)")


if __name__ == "__main__":
    main()

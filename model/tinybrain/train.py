#!/usr/bin/env python3
"""train.py - distil v4m's goal distribution into a small MLP (tinybrain phase 1).

  python3 model/tinybrain/train.py --data model/out/tb_labels.jsonl model/out/tb_labels_panel.jsonl \
      --out model/out/tinybrain.bin

Input: label_states.py output ({"state", "p": 8 goal probs, "u": 10 urgency probs}).
Net: 56 features -> H1 -> H2 -> 8 goal logits + 10 urgency logits (ReLU). Loss:
soft cross-entropy on both heads (the WHOLE distribution, not the argmax, so
the device can keep sampling and hesitating). numpy only; a 120K-state run
takes a few minutes on a laptop CPU.

Export (tinybrain.bin, read by common/llm/tiny_brain.c):
  "TBRN" | u32 version (1) | u32 n_in, n_h1, n_h2, n_goal, n_urg
  then per layer (h1, h2, goal, urg): int8 W[rows][cols] row-major,
  f32 scale[rows] (W_float = W_int8 * scale), f32 bias[rows]
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import N_FEAT, features  # noqa: E402

MAGIC, VERSION = b"TBRN", 1


def load(paths):
    X, P, U, S = [], [], [], []
    for ln in (ln for p in ([paths] if isinstance(paths, str) else paths) for ln in open(p)):
        o = json.loads(ln)
        S.append(o["state"]); X.append(features(o["state"])); P.append(o["p"]); U.append(o["u"])
    return np.array(X, np.float32), np.array(P, np.float32), np.array(U, np.float32), S


def init(rng, n_in, h1, h2):
    def lin(i, o):
        return [rng.normal(0, math.sqrt(2.0 / i), (o, i)).astype(np.float32), np.zeros(o, np.float32)]
    return lin(n_in, h1) + lin(h1, h2) + lin(h2, 8) + lin(h2, 10)


def forward(params, X):
    W1, b1, W2, b2, Wg, bg, Wu, bu = params
    z1 = X @ W1.T + b1; a1 = np.maximum(z1, 0)
    z2 = a1 @ W2.T + b2; a2 = np.maximum(z2, 0)
    return a1, a2, a2 @ Wg.T + bg, a2 @ Wu.T + bu


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def grads(params, X, P, U, w_urg):
    W1, b1, W2, b2, Wg, bg, Wu, bu = params
    a1, a2, lg, lu = forward(params, X)
    n = len(X)
    qg, qu = softmax(lg), softmax(lu)
    loss = -(P * np.log(qg + 1e-9)).sum(1).mean() - w_urg * (U * np.log(qu + 1e-9)).sum(1).mean()
    dg, du = (qg - P) / n, w_urg * (qu - U) / n
    gWg, gbg, gWu, gbu = dg.T @ a2, dg.sum(0), du.T @ a2, du.sum(0)
    da2 = (dg @ Wg + du @ Wu) * (a2 > 0)
    gW2, gb2 = da2.T @ a1, da2.sum(0)
    da1 = (da2 @ W2) * (a1 > 0)
    gW1, gb1 = da1.T @ X, da1.sum(0)
    return loss, [gW1, gb1, gW2, gb2, gWg, gbg, gWu, gbu]


def quantize(params):
    """int8 per output row, symmetric; returns (q, scale) per weight matrix"""
    out = []
    for W in params[0::2]:
        s = np.abs(W).max(1) / 127.0; s[s == 0] = 1.0
        out.append((np.clip(np.round(W / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32)))
    return out


def dequant(params, q):
    p = list(params)
    for k, (qi, s) in enumerate(q):
        p[2 * k] = qi.astype(np.float32) * s[:, None]
    return p


def metrics(params, X, P, U):
    _, _, lg, lu = forward(params, X)
    qg, qu = softmax(lg), softmax(lu)
    t1 = P.argmax(1); s1 = qg.argmax(1)
    s_top2 = np.argsort(-qg, 1)[:, :2]
    kl = (P * (np.log(P + 1e-9) - np.log(qg + 1e-9))).sum(1).mean()
    tv = 0.5 * np.abs(P - qg).sum(1).mean()
    # what a device would see: P(sampled student goal == sampled teacher goal)
    match = (P * qg).sum(1).mean() / (P * P).sum(1).mean()
    exp_u_t = (U * np.arange(10)).sum(1); exp_u_s = (qu * np.arange(10)).sum(1)
    return {"top1": float((t1 == s1).mean()), "teacher_top1_in_student_top2":
            float((s_top2 == t1[:, None]).any(1).mean()), "kl": float(kl), "tv": float(tv),
            "sample_match_vs_self": float(match), "urgency_mae": float(np.abs(exp_u_t - exp_u_s).mean())}


def export(path, params, q, h1, h2):
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<6I", VERSION, N_FEAT, h1, h2, 8, 10))
        for k, (qi, s) in enumerate(q):
            f.write(qi.tobytes()); f.write(s.tobytes()); f.write(params[2 * k + 1].astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs="+", default=["model/out/tb_labels.jsonl"])
    ap.add_argument("--out", default="model/out/tinybrain.bin")
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--w-urg", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None, help="write final metrics as JSON here")
    args = ap.parse_args()

    X, P, U, _ = load(args.data)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X)); nv = int(len(X) * args.val_frac)
    va, tr = idx[:nv], idx[nv:]
    params = init(rng, N_FEAT, args.h1, args.h2)
    n_params = sum(p.size for p in params)
    print(f"{len(tr)} train / {len(va)} val states; net {N_FEAT}-{args.h1}-{args.h2}-(8+10), {n_params} params")
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    step, steps = 0, args.epochs * math.ceil(len(tr) / args.batch)
    best, best_params = 1e9, None
    for ep in range(args.epochs):
        rng.shuffle(tr); tot = 0.0
        for b in range(0, len(tr), args.batch):
            bi = tr[b:b + args.batch]
            loss, g = grads(params, X[bi], P[bi], U[bi], args.w_urg)
            tot += loss * len(bi); step += 1
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / steps))     # cosine to 0
            for k in range(len(params)):
                if k % 2 == 0:
                    g[k] = g[k] + args.wd * params[k]
                m[k] = 0.9 * m[k] + 0.1 * g[k]; v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                mh, vh = m[k] / (1 - 0.9 ** step), v[k] / (1 - 0.999 ** step)
                params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        vm = metrics(params, X[va], P[va], U[va])
        if vm["kl"] < best:
            best, best_params = vm["kl"], [p.copy() for p in params]
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"epoch {ep + 1:3d} train loss {tot / len(tr):.4f} | val top1 {vm['top1']:.3f} "
                  f"KL {vm['kl']:.4f} TV {vm['tv']:.3f} urgMAE {vm['urgency_mae']:.2f}")
    params = best_params
    q = quantize(params)
    fm, qm = metrics(params, X[va], P[va], U[va]), metrics(dequant(params, q), X[va], P[va], U[va])
    print("val, float :", json.dumps({k: round(x, 4) for k, x in fm.items()}))
    print("val, int8  :", json.dumps({k: round(x, 4) for k, x in qm.items()}))
    export(args.out, params, q, args.h1, args.h2)
    print(f"-> {args.out} ({os.path.getsize(args.out)} bytes)")
    if args.report:
        json.dump({"n_params": int(n_params), "bytes": os.path.getsize(args.out), "float": fm, "int8": qm,
                   "args": vars(args)}, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()

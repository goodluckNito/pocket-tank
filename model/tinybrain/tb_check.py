#!/usr/bin/env python3
"""tb_check.py - the C tinybrain (common/llm/tiny_brain.c, via tb_check) and the
Python twin (features.py + the int8 weights of the same file) must give the
same logits on the same state lines. Also rejects a v3 line and a garbled one.

  python3 model/tinybrain/tb_check.py model/out/tinybrain.bin states.txt"""
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import features  # noqa: E402


def load_bin(path):
    b = open(path, "rb").read()
    assert b[:4] == b"TBRN"
    _, n_in, h1, h2, ng, nu = struct.unpack("<6I", b[4:28])
    p, layers = 28, []
    for i, o in ((n_in, h1), (h1, h2), (h2, ng), (h2, nu)):
        W = np.frombuffer(b, np.int8, i * o, p).reshape(o, i).astype(np.float32); p += i * o
        s = np.frombuffer(b, np.float32, o, p); p += 4 * o
        bias = np.frombuffer(b, np.float32, o, p); p += 4 * o
        layers.append((W * s[:, None], bias))
    assert p == len(b)
    return layers


def forward(layers, x):
    (W1, b1), (W2, b2), (Wg, bg), (Wu, bu) = layers
    a1 = np.maximum(W1 @ x + b1, 0); a2 = np.maximum(W2 @ a1 + b2, 0)
    return np.concatenate([Wg @ a2 + bg, Wu @ a2 + bu])


def main():
    bin_path, states_path = sys.argv[1], sys.argv[2]
    exe = os.path.join(HERE, "tb_check")
    states = [s for s in open(states_path).read().splitlines() if s]
    bad = ["zone 2 hunger 7 energy 5 stress 2 curiosity 8 bold 4 social 6 stage adult trust 5 "
           "food near 12 shadow none friend mid 3 bubble mid 10 reef far 7 wall clear last explore time day",
           "zone 2 hunger x"]
    out = subprocess.run([exe, bin_path], input="\n".join(states + bad) + "\n",
                         capture_output=True, text=True, check=True)
    rows = out.stdout.splitlines()
    assert rows[-2:] == ["BAD", "BAD"], f"C accepted a non-v4 line: {rows[-2:]}"
    layers = load_bin(bin_path)
    worst = 0.0
    for s, r in zip(states, rows):
        c = np.array([float(v) for v in r.split()])
        worst = max(worst, float(np.abs(c - forward(layers, np.array(features(s), np.float32))).max()))
    print(f"{len(states)} states: max |C - python| logit difference {worst:.2e}; {out.stderr.strip()}")
    sys.exit(0 if worst < 1e-3 else 1)


if __name__ == "__main__":
    main()

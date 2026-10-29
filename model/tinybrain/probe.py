#!/usr/bin/env python3
"""probe.py - the v4m acceptance panel (docs/stats.md, "Schema v4 data cycle"),
asked of BOTH brains: the shipped transformer (via label_q4) and a tinybrain file.

  python3 model/tinybrain/probe.py --tb model/out/tinybrain.bin

The probes are probe_dist.py's synthetic v4 states. A tinybrain that matches
v4m here keeps every behaviour the project tuned for (boredom, the personality
cliffs, starvation, night rest); this is the gate before it replaces v4m.
Prints a markdown table; --json writes the numbers too."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from features import GOALS, features  # noqa: E402
from tb_check import forward, load_bin  # noqa: E402

G = {g: i for i, g in enumerate(GOALS)}


def S(zone=2, hunger=3, energy=6, stress=1, curiosity=6, bold=5, social=5, stage="adult",
      food="none", friend="mid 3", bubble="mid 10", reef="far 7", wall="clear", last="explore",
      time="day", trust=5, bored=1):
    return (f"zone {zone} hunger {hunger} energy {energy} stress {stress} curiosity {curiosity} "
            f"bold {bold} social {social} stage {stage} trust {trust} bored {bored} food {food} "
            f"friend {friend} bubble {bubble} reef {reef} wall {wall} last {last} time {time}")


CONTENT = [dict(hunger=2, energy=7, stress=0, curiosity=7), dict(hunger=3, energy=9, stress=1, curiosity=3),
           dict(hunger=1, energy=8, stress=0, curiosity=5, friend="near 2"),
           dict(hunger=2, energy=6, stress=2, curiosity=8, zone=5, reef="mid 10", bubble="far 3"),
           dict(hunger=4, energy=9, stress=0, curiosity=6, last="dart_play"),
           dict(hunger=0, energy=9, stress=0, curiosity=9, zone=1, reef="near 12", bubble="far 6")]


def panel():
    """name -> (list of states, reducer over their prob vectors)"""
    P = {}
    for b in (0, 3, 5, 6, 9):
        st = [S(hunger=2, energy=7, stress=0, curiosity=7, bubble="near 12", last="visit_bubbles", bored=b)]
        P[f"bored {b} at the bubbles: P(bubbles again)"] = (st, lambda p: p[0][G["visit_bubbles"]])
        P[f"bored {b} at the bubbles: P(explore)"] = (st, lambda p: p[0][G["explore"]])
    for b in (0, 5, 9):
        P[f"social 8 following, bored {b}: P(follow)"] = (
            [S(hunger=2, energy=7, stress=0, curiosity=6, social=8, friend="near 12", last="follow_friend", bored=b)],
            lambda p: p[0][G["follow_friend"]])
    P["night elder, bored 9, last rest: P(rest)"] = (
        [S(hunger=2, energy=5, stress=0, time="night", stage="elder", last="rest", bored=9)], lambda p: p[0][G["rest"]])
    P["starving + bored 9: P(seek_food)"] = (
        [S(hunger=9, food="near 12", bored=9, last="seek_food")], lambda p: p[0][G["seek_food"]])
    ids = [S(hunger=9, food="near 12", bold=b, social=s, stage=st)
           for b in (0, 5, 9) for s in (0, 9) for st in ("fry", "adult", "elder")]
    P["starving, min P(seek_food) over 18 identities"] = (ids, lambda p: min(q[G["seek_food"]] for q in p))
    P["starving + friend none: P(seek_food)"] = (
        [S(hunger=9, food="near 12", friend="none")], lambda p: p[0][G["seek_food"]])
    P["social 0, mean P(follow) over 6 content states"] = (
        [S(social=0, **c) for c in CONTENT], lambda p: float(np.mean([q[G["follow_friend"]] for q in p])))
    P["social 9, mean P(follow) over 6 content states"] = (
        [S(social=9, **c) for c in CONTENT], lambda p: float(np.mean([q[G["follow_friend"]] for q in p])))
    P["bold 0, mean P(dart) over 6 content states"] = (
        [S(bold=0, **c) for c in CONTENT], lambda p: float(np.mean([q[G["dart_play"]] for q in p])))
    P["bold 9, mean P(dart) over 6 content states"] = (
        [S(bold=9, **c) for c in CONTENT], lambda p: float(np.mean([q[G["dart_play"]] for q in p])))
    P["elder at night, content: P(rest)"] = (
        [S(hunger=2, energy=5, stress=0, time="night", stage="elder")], lambda p: p[0][G["rest"]])
    P["friend none, content: P(follow) (impossible)"] = (
        [S(hunger=2, energy=7, stress=0, curiosity=7, social=9, friend="none")], lambda p: p[0][G["follow_friend"]])
    allst = sorted({s for sts, _ in P.values() for s in sts})
    P["max P(flee_shadow) over the panel"] = (allst, lambda p: max(q[G["flee_shadow"]] for q in p))
    return P


def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tb", nargs="+", required=True, help="tinybrain file(s) to compare")
    ap.add_argument("--model-bin", default=os.path.join(MODEL_DIR, "out", "model_q4.bin"))
    ap.add_argument("--tok-bin", default=os.path.join(MODEL_DIR, "out", "tokenizer.bin"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    P = panel()
    states = sorted({s for sts, _ in P.values() for s in sts})
    out = subprocess.run([os.path.join(HERE, "label_q4"), args.model_bin, args.tok_bin],
                         input="\n".join(states) + "\n", capture_output=True, text=True, check=True)
    v4m = {o["state"]: o["p"] for o in map(json.loads, out.stdout.splitlines())}
    brains = {"v4m (14.3M)": v4m}
    for path in args.tb:
        L = load_bin(path)
        brains[os.path.basename(path)] = {s: softmax(forward(L, np.array(features(s), np.float32))[:8]).tolist()
                                          for s in states}
    names = list(brains)
    print("| probe | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    res = {}
    for k, (sts, red) in P.items():
        row = [float(red([brains[n][s] for s in sts])) for n in names]
        res[k] = dict(zip(names, row))
        print(f"| {k} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    worst = {n: max(abs(res[k][n] - res[k][names[0]]) for k in res) for n in names[1:]}
    print("\nlargest gap to v4m on any probe: " + ", ".join(f"{n} {v:.2f}" for n, v in worst.items()))
    if args.json:
        json.dump(res, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()

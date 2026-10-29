#!/usr/bin/env python3
"""label_states.py - sample schema-v4 states and label them with the shipped model's
full goal distribution (tinybrain phase 1: v4m becomes the teacher).

  python3 model/tinybrain/label_states.py --count 120000 --out model/out/tb_labels.jsonl

States come from three sources:
  - gen_traces.py's headless tank with its edge-case perturbations (the same
    distribution v4m was trained on),
  - --random-frac of independently-uniform field combinations, for coverage of
    corners the sim rarely visits (the student must not fall apart there), and
  - --panel-frac of states near probe.py's templates: each field of the
    template is redrawn with probability 1/2. The panel asks about states the
    sim geometry never makes (zone 2 with the bubbles near); without these the
    student and v4m extrapolate differently there. probe.py's exact states are
    never labeled, so the panel stays a held-out gate.
Duplicates are dropped, then the batch is split over --workers label_q4
processes (~110 ms per state per core on a laptop CPU).

Stdlib only. Build the labeler first (see label_q4.c)."""

import argparse
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(HERE)
sys.path.insert(0, MODEL_DIR)
import gen_traces as gt  # noqa: E402

DISTS = ["near", "mid", "far"]
STAGES = ["fry", "juv", "adult", "elder"]


def sim_states(rng, count, perturb=0.45, ticks_between=40, repopulate=250):
    """gen_traces.py --dry-run, in-process (schema 4)"""
    gt.SCHEMA = 4
    tank = gt.Tank(rng)
    for _ in range(200):
        tank.tick()
    out, fish_idx = [], 0
    while len(out) < count:
        if out and len(out) % repopulate == 0 and fish_idx % len(tank.fish) == 0:
            tank = gt.Tank(rng)
            for _ in range(200):
                tank.tick()
        for _ in range(ticks_between):
            tank.tick()
        f = tank.fish[fish_idx % len(tank.fish)]
        fish_idx += 1
        if rng.random() < perturb:
            tank.perturb(f)
        out.append(gt.encode(tank, f))
    return out


def sighting(rng, p_none):
    if rng.random() < p_none:
        return "none"
    return f"{rng.choice(DISTS)} {rng.randint(1, 12)}"


# field order of the v4 line; probe.py's template defaults (sightings keep "dist clock")
TEMPLATE = dict(zone="2", hunger="3", energy="6", stress="1", curiosity="6", bold="5", social="5",
                stage="adult", trust="5", bored="1", food="none", friend="mid 3", bubble="mid 10",
                reef="far 7", wall="clear", last="explore", time="day")


def line(fields):
    return " ".join(f"{k} {fields[k]}" for k in TEMPLATE)


def random_fields(rng):
    """every field drawn independently (bubble and reef are always somewhere in view)"""
    f = {k: str(rng.randint(0, 9)) for k in ("hunger", "energy", "stress", "curiosity", "bold",
                                             "social", "trust", "bored")}
    f.update(zone=str(rng.randint(1, 6)), stage=rng.choice(STAGES), food=sighting(rng, 0.35),
             friend=sighting(rng, 0.15), bubble=sighting(rng, 0.0), reef=sighting(rng, 0.0),
             wall="clear" if rng.random() < 0.5 else f"{rng.choice(DISTS[:2])} {rng.randint(1, 12)}",
             last=rng.choice(gt.GOALS_V4), time="night" if rng.random() < 0.25 else "day")
    return f


def random_state(rng):
    return line(random_fields(rng))


def panel_state(rng):
    """probe.py's template with each field redrawn with probability 1/2"""
    fresh = random_fields(rng)
    return line({k: fresh[k] if rng.random() < 0.5 else d for k, d in TEMPLATE.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", type=int, required=True, help="states to label (before dedup)")
    ap.add_argument("--random-frac", type=float, default=0.25)
    ap.add_argument("--panel-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--model-bin", default=os.path.join(MODEL_DIR, "out", "model_q4.bin"))
    ap.add_argument("--tok-bin", default=os.path.join(MODEL_DIR, "out", "tokenizer.bin"))
    ap.add_argument("--labeler", default=os.path.join(HERE, "label_q4"))
    ap.add_argument("--out", default=os.path.join(MODEL_DIR, "out", "tb_labels.jsonl"))
    args = ap.parse_args()
    if not os.path.exists(args.labeler):
        sys.exit(f"no labeler at {args.labeler} - build label_q4.c first")

    rng = random.Random(args.seed)
    n_rand = int(args.count * args.random_frac)
    n_panel = int(args.count * args.panel_frac)
    states = (sim_states(rng, args.count - n_rand - n_panel) + [random_state(rng) for _ in range(n_rand)]
              + [panel_state(rng) for _ in range(n_panel)])
    sys.path.insert(0, HERE)
    from probe import panel                       # the gate's own states stay unlabeled
    held_out = {s for sts, _ in panel().values() for s in sts}
    states = [s for s in dict.fromkeys(states) if s not in held_out]   # dedup, order kept
    rng.shuffle(states)
    print(f"{len(states)} distinct states ({args.count} drawn); labeling on {args.workers} workers",
          file=sys.stderr)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        procs = []
        for w in range(args.workers):
            chunk = states[w::args.workers]
            src, dst = os.path.join(tmp, f"in{w}.txt"), os.path.join(tmp, f"out{w}.jsonl")
            with open(src, "w") as fh:
                fh.write("\n".join(chunk) + "\n")
            procs.append((subprocess.Popen([args.labeler, args.model_bin, args.tok_bin],
                                           stdin=open(src), stdout=open(dst, "w")), dst))
        for p, _ in procs:
            if p.wait() != 0:
                sys.exit("a labeler worker failed")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as out:
            for _, dst in procs:
                out.write(open(dst).read())
    print(f"done: {len(states)} labeled -> {args.out} in {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()

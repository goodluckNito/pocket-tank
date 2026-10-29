"""features.py - the schema-v4 state line as a fixed-length float vector.

The byte-exact twin of tb_features() in common/llm/tiny_brain.c: change one,
change both (tb_check.py compares them). Phase 1 deliberately reads the SAME
bucketed line the LLM reads, so the small network can be distilled from v4m
on identical inputs; continuous inputs are a later step (docs/tinybrain.md).

Layout (N_FEAT = 56):
   0- 5  zone one-hot (1..6)
   6-13  hunger energy stress curiosity bold social trust bored, each / 9
  14-17  stage one-hot (fry juv adult elder)
  18-41  food friend bubble reef, 6 each: present, near, mid, far, sin, cos
         (clock as an angle, 12 = dead ahead = (0, 1); all zero when none)
  42-46  wall: clear, near, mid, sin, cos
  47-54  last one-hot (GOALS order)
  55     night
"""
import math

GOALS = ["seek_food", "flee_shadow", "visit_bubbles", "follow_friend",
         "explore", "rest", "dart_play", "inspect_reef"]
STAGES = ["fry", "juv", "adult", "elder"]
DRIVES = ["hunger", "energy", "stress", "curiosity", "bold", "social", "trust", "bored"]
SIGHTINGS = ["food", "friend", "bubble", "reef"]
DISTS = ["near", "mid", "far"]
N_FEAT = 56


def _clock(c):
    a = int(c) / 12.0 * 2.0 * math.pi
    return math.sin(a), math.cos(a)


def features(state):
    w = state.split()
    kv, i = {}, 0
    while i < len(w):                     # key value [clock]
        k = w[i]
        if k in SIGHTINGS or k == "wall":
            if w[i + 1] in ("none", "clear"):
                kv[k] = None; i += 2
            else:
                kv[k] = (w[i + 1], w[i + 2]); i += 3
        else:
            kv[k] = w[i + 1]; i += 2
    x = [0.0] * N_FEAT
    x[int(kv["zone"]) - 1] = 1.0
    for j, d in enumerate(DRIVES):
        x[6 + j] = int(kv[d]) / 9.0
    x[14 + STAGES.index(kv["stage"])] = 1.0
    for j, s in enumerate(SIGHTINGS):
        o = 18 + 6 * j
        if kv[s] is not None:
            dist, c = kv[s]
            x[o] = 1.0
            x[o + 1 + DISTS.index(dist)] = 1.0
            x[o + 4], x[o + 5] = _clock(c)
    if kv["wall"] is None:
        x[42] = 1.0
    else:
        dist, c = kv["wall"]
        if dist == "far":
            raise ValueError("wall far is not a v4 value (far reads clear)")
        x[43 + DISTS.index(dist)] = 1.0
        x[45], x[46] = _clock(c)
    x[47 + GOALS.index(kv["last"])] = 1.0
    x[55] = 1.0 if kv["time"] == "night" else 0.0
    return x

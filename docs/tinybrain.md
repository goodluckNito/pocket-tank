# tinybrain roadmap — the LLM teaches, a small network decides

**Status: proposal. Phase 1 tooling is written (sim only); no model is
trained or shipped. The 14M transformer is still what runs.**

## The idea in one paragraph

Each decision reads about 25 discrete fields (drives 0–9, bucketed distances,
clock directions, stage, time of day) and picks one of 7 goals plus an
urgency digit. That is a classification problem, but the 14M transformer
answers it one token at a time: ~3.7 s and 7.56 MB of flash per decision on
the board. A plain ~10 KB network on the same inputs answers in
microseconds. The runtime model isn't the bottleneck anyway: the teacher
agrees with itself only 82% of the time and v4m already reaches 72–75%
(docs/stats.md). So the LLM moves out of the device and into the training
pipeline. The compute and memory that frees up pays for what a frozen shared
model can't do: per-fish memory, learning from experience, and inheritance.

**Trade-off:** the headline "a tiny language model on an $8 chip" goes. The
fish are still taught by a language model; they just don't run one.

## At a glance

| # | phase | what you get | done when | size |
|---|---|---|---|---|
| 1 | Distil v4m into a small MLP, in the sim | same behaviour, µs decisions, 10–30 KB | acceptance panel and 5-min census match v4m | small: tooling exists, ~1 h labeling + minutes of training |
| 2 | Put it on the board | free core 1, 8 MB flash, 1.5 MB PSRAM; battery gain; no "starving-ignored" | hardware census + batlog | small–medium |
| 3 | Train against the real teacher cheaply | better than v4m; schema changes take minutes | `eval.py --teacher` ≥ v4m's 72% | medium |
| 4 | Organic fish: memory, on-device learning, inheritance | individuals that diverge with experience | survival panel holds across a mutated population; it *looks* more alive | large |
| 5 | LLM as critic and designer | tuning toward "a lively tank", not single labels | critic preferences agree with a human | medium, open-ended |

Phases 1→2 are the critical path. Phase 3 can start any time after 1.
Phase 4 needs 2 (device budget) and benefits from 5 (a way to judge it).

---

## Phase 1 — distil v4m into an MLP (sim only)

**Goal:** a drop-in replacement that behaves like v4m.

**Approach:** v4m is the teacher. It already gives a full goal distribution
per state (what `probe_dist.py` reads), and training on the distribution
keeps sampling and hesitation intact. No new gemma data needed.

**Already written (this branch):**

| piece | file |
|---|---|
| labeler: the shipped 4-bit engine, outputs P(goal) and P(urgency \| greedy goal) | `model/tinybrain/label_q4.c` |
| state sampler (gen_traces' tank + perturbations, uniform-random states, probe-template neighborhood) with a multi-core fan-out | `model/tinybrain/label_states.py` |
| the v4 line as 56 floats | `model/tinybrain/features.py`, twin of `tb_features()` in C |
| numpy trainer (soft cross-entropy on both heads), int8 export | `model/tinybrain/train.py` |
| C runtime, weights read in place, no allocation | `common/llm/tiny_brain.{h,c}` |
| C ↔ Python parity check and timing | `model/tinybrain/tb_check.{c,py}` |
| v4m acceptance panel (docs/stats.md), asked of both brains | `model/tinybrain/probe.py` |
| LightGBM baseline | `model/tinybrain/baselines.py` |
| `advisor_core` picks the engine by the blob's magic (`TBRN`); sim `POCKET_BRAIN=` and `POCKET_ADVISOR_MS=` (device-latency emulation) | `common/llm/advisor_core.c`, `sim/advisor_llm.c` |
| CI: parity + `--selftest-llm` with the tinybrain, skipped until `model/out/tinybrain.bin` is committed | `.github/workflows/checks.yml` |

The reflex layer, hesitation and progression code don't change: the same
distribution layer (sampled goal, confidence, runner-up, damaged-model check)
runs over either engine.

**Steps (on your machine):**

```bash
cc -O3 -std=c11 -Icommon/llm -o model/tinybrain/label_q4 \
  model/tinybrain/label_q4.c common/llm/q4_model.c common/llm/word_tok.c -lm
cc -O2 -std=c11 -Icommon/llm -o model/tinybrain/tb_check \
  model/tinybrain/tb_check.c common/llm/tiny_brain.c -lm
# 1. label: ~110 ms/state/core, so 120K states is ~1 h on 4 cores (less on an M4 Pro)
python3 model/tinybrain/label_states.py --count 120000 --seed 7
python3 model/tinybrain/label_states.py --count 24000 --random-frac 0 --panel-frac 1 \
  --seed 11 --out model/out/tb_labels_panel.jsonl
# 2. train (seconds to minutes, numpy only), try --h1/--h2 64 and 128
python3 model/tinybrain/train.py --data model/out/tb_labels.jsonl model/out/tb_labels_panel.jsonl
# 3. gates
python3 model/gen_traces.py --count 500 --dry-run --schema 4 --seed 3 > /tmp/s.txt
python3 model/tinybrain/tb_check.py model/out/tinybrain.bin /tmp/s.txt
python3 model/tinybrain/probe.py --tb model/out/tinybrain.bin
cd sim && POCKET_BRAIN=../model/out/tinybrain.bin ./fishsim --selftest-llm 5
cd sim && POCKET_ADVISOR_MS=2500 ./fishsim --selftest-llm 5          # v4m at device speed, for comparison
```

**Acceptance:**

- `probe.py`: every panel row within ~0.15 of v4m. The survival and night
  rows matter most; the social / fresh-fish rows can differ where v4m itself
  is extrapolating (see the first look below).
- 5-minute census (`--selftest-llm 5`) against v4m at `POCKET_ADVISOR_MS=2500`:
  goal shares, landmark time, crowding and zones per fish-minute within
  run-to-run noise. Run each brain on 2–3 seeds, since one 60 s run is far
  too noisy to judge.
- Commit `model/out/tinybrain.bin` (`git add -f`, as for `model_q4.bin`) so
  CI runs the parity and selftest step.

**First look (one short exploratory pass, not a result):**

| | v4m | tinybrain 64-64 | tinybrain 128-128 | LightGBM 100×15 | LightGBM 400×31 |
|---|---|---|---|---|---|
| size | 7.56 MB | 10 KB | 28 KB | ~180 KB | ~1.5 MB |
| decision, laptop CPU | ~113 ms | ~8 µs | — | — | — |
| top-1 = v4m's top-1 (held-out) | — | 88% | 90% | 89%\* | 92%\* |
| v4m's top-1 in student top-2 | — | 98% | 98% | 97%\* | 99%\* |

\*trees measured on a 57K subset; the MLPs on the full 120K, int8 weights.
C and Python agree to 1e-6. Training took seconds.

What the first look already says:

- On the sim's own state distribution the MLP matches v4m closely, e.g.
  P(follow | social ≥ 8, content) 0.48 vs 0.48, and P(bubbles kept | fresh,
  bubble near) 0.61 vs 0.60.
- **The panel disagreed** on two rows: the social-8 follow cliff (v4m 0.84,
  tinybrain 0.4–0.6) and "a fresh fish keeps the bubbles" (0.21 vs 0.5–0.6).
  Those probe states can't happen in the sim (zone 2 with the bubbles
  "near", while the column stands in zones 3/6), so both models are
  extrapolating there. The `--panel-frac` sampler exists for exactly this and
  is the next thing to run. It keeps the panel's own states out of training,
  so the panel stays a held-out gate.
- Starvation (min P(seek_food) over 18 identities 0.94–0.97 vs 0.85), night
  rest, boredom at 6–9, the bold cliff, and P(flee) ≈ 0 already match.
- Trees beat the MLP per sample but not per byte. They're the upper bound to
  watch, not the thing to ship.

## Phase 2 — on the board

1. **Firmware init.** `advisor_llm_esp.c` assumes a q4 model after init (the
   kernel bench, the `dim/layers` log line); guard both with
   `advisor_core_tiny()`. `advisor_llm_esp_init` passes the whole partition
   size, and `tb_open` already accepts bytes past the last layer.
2. **Model trailer / OTA.** `model_trailer.c` hashes the partition body
   against `PT_MODEL_SHA256` and raises the "unknown body" notice otherwise.
   The build has to know the tinybrain's hash and tag, and `docs/OTA.md`'s
   in-place model write gets much simpler for a 10 KB body.
3. **Scheduling for a near-free brain.** Drop the one-slot worker task and
   the rotation, and ask each fish synchronously in `tank_tick`. Re-ask
   gating then becomes a design knob ("how often does a fish change its
   mind"), not a compute budget.
4. **Reclaim memory.** The 8 MB model partition and the 1.5 MB KV cache
   (`docs/memory_budget.md`). Decide what they're for: assets, scene cache,
   OTA headroom, phase-4 state.
5. **Measure:** µs per decision on the S3, a batlog on battery with core 1
   idle, and the starving-ignored count in a long run.

**Keep v4m as a one-flash fallback** while this settles. The magic-byte
switch means either file works in the same partition.

## Phase 3 — learn from the real teacher, cheaply

Phase 1's ceiling is v4m, which is itself a student. Cut it out:

- **Soft teacher labels:** sample gemma k times per state at T=0.7, or read
  token logprobs if the server exposes them. Mix with v4m labels at first.
- **Active learning:** ask the teacher only where the tinybrain is unsure
  (high entropy, or an ensemble disagrees). An overnight run of ~20K uniform
  states becomes a few thousand targeted ones.
- **Schema changes become columns:** the queued `shrimp N out`
  (`docs/retrain-next.md`) is one new input feature, not a new tokenizer and
  a 30-minute retrain.
- **Continuous inputs:** exact distance, bearing and real-valued drives from
  `fish_t` instead of text buckets, so behaviour stops jumping at bucket
  edges. This breaks parity with the LLM's line, so it waits until v4m labels
  aren't needed.
- **Hard rules where the teacher is wrong:** v4m gives P(follow) ≈ 0.25 to a
  social fish with `friend none`, which is impossible. A logit mask fixes it
  exactly.
- **Real states:** label what the device actually sees (firmware `ESP_LOGI`
  state lines, or a sim `--dump-states` flag), not only gen_traces' tank.

**Acceptance:** `eval.py --teacher` on the tinybrain ≥ v4m's 72% (n = 60,
seed 777). The phase-1 gates still hold.

## Phase 4 — the organic part

Organic behaviour doesn't come from parameter count. It comes from fish being
stochastic, remembering, and changing because of what happened to them. A
frozen 14M model shared by every fish only does the first.

- **4a Learning on the device (start here).** Each fish gets a small
  personal layer: a bias or low-rank adapter on the goal logits, saved in NVS
  with the fish. It updates from simple rewards: ate after seeking food,
  startled by a tap, found nothing at the reef. A fish that keeps finding
  food at the reef drifts there; one that gets tapped a lot turns wary. Two
  fish with identical traits become different individuals. Cheap, and
  visible.
- **4b Inheritance.** A fry gets a blend of its parents' personal layers
  with small mutations. That fits the existing arrivals and life stages
  (`docs/progression.md`).
- **4c Memory.** A small recurrent state per fish (a GRU cell of 16–32 floats)
  fed back into the network gives moods and momentum. Hand-built counters
  like `bored` could become learned. This needs whole-trajectory training in
  the sim, so it comes after phase 5's rollout tooling.
- **Guardrails:** the shared network stays frozen and personal layers are
  clamped. The survival rows of the panel are checked over a population of
  mutated or experienced fish, not just the base network.

**Open design questions for you:** which events count as reward, and how
visible drift should be (a card line like "Mira has learned the reef is
where food falls"?). Should learned habits survive a sale/rebuy, and pass to
fry?

## Phase 5 — the LLM as critic and designer

- **Critic:** run the headless sim for 60 s with two candidate brains or
  learning rates, give the LLM both behaviour logs (census lines, goal
  timelines), and ask which tank looks more alive. Use the preferences to
  tune or evolve the network and the phase-4 learning rates in the PC sim.
  This targets the actual goal instead of matching single decisions.
  Calibrate against your own judgment on a handful of pairs first.
- **Designer:** have it draft features, reward functions and probe panels in
  the style of `docs/stats.md` and `prompt_check.py`.

## Risks

- **No gemma agreement number yet.** Phase 1 is measured against v4m only;
  phase 3 is where it gets checked against the teacher.
- **The panel's out-of-distribution states.** See phase 1. Decide whether
  the gate should be rewritten with geometrically possible states, or kept
  and trained toward.
- **Urgency is conditioned on the greedy goal**, so it's slightly off when a
  sampled goal isn't the mode. A per-goal urgency head (8×10) fixes it if it
  matters.
- **The README and video story change.** Worth deciding before phase 2 ships.

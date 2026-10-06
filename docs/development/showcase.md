# Reproducible demo pipeline

This directory produces README animations from recorded model decisions. It keeps the
showcase separate from training while reusing the exact Vision-Jev prompt, backbone,
LoRA adapter, calibrated decision heads, and environment packages.

## Comparison contract

Each example is evaluated twice per parameter group:

| Group | Left panel | Right panel |
| --- | --- | --- |
| 0.8B | Original Qwen3.5-0.8B | Vision-Jev-0.8B |
| 9B | Original Qwen3.5-9B | Vision-Jev-9B |

Each visual example freezes exactly one unseen seed. Both panels receive the same
rendered state, mission, candidates, environment seed,
maximum step count, and greedy decoding policy. Baseline parse failures are shown as
failures; the recorder never substitutes an oracle action. Every trajectory stores frame
hashes, actions, calibrated probabilities when available, per-step latency, and outcome.

The initial suite covers FourRooms, DoorKey, LavaCrossing, Dynamic Obstacles, and BabyAI.
Procgen Maze and Boxoban should use the same trajectory contract through isolated
environment adapters, because Procgen has a pinned Python 3.10 runtime and Boxoban uses
the repository's custom level parser.

## Run

Install both inference and interactive-environment dependencies in the same environment,
then set the external data root:

```bash
pip install -e '.[train,rl]'
export DATA_ROOT=/data/vision-jev
python -m showcase.cli validate
```

Run the complete suite sequentially. Only one model is resident on the GPU at a time;
existing complete trajectories are skipped when the command is resumed:

```bash
python -m showcase.cli run-all
```

Record one deterministic episode:

```bash
python -m showcase.cli record \
  --example fourrooms-navigation \
  --model qwen35-08b-base
```

After recording both models in a parameter group, render the comparison:

```bash
python -m showcase.cli render \
  --baseline showcase/output/fourrooms-navigation/qwen35-08b-base/trajectory.json \
  --trained showcase/output/fourrooms-navigation/vision-jev-08b/trajectory.json \
  --output asset/demos/fourrooms-navigation/0.8b.gif
```

Only reviewed GIFs and their final metric summaries belong in `asset/demos/`. Raw frames
and trajectories remain under the ignored `showcase/output/` directory. The README
fragment generator refuses to create markup until every configured GIF exists:

```bash
python -m showcase.cli readme --output showcase/output/README-demos.md
```

After reviewing the generated fragment, update the marker-delimited section in the main
README without manual copying:

```bash
python -m showcase.cli publish-readme
```

## Selection rules

- Reserve exactly one unseen seed per visual example before running any model.
- Use multi-seed evaluation only in the formal test protocol, not inside a README animation.
- Keep failed runs; do not search seeds independently for Vision-Jev and Qwen.
- Publish all configured examples or disclose the deterministic selection rule.
- Report success, decisions, cumulative model latency, and invalid output failures.
- Preserve the trajectory JSON and exact model/config revisions for release evidence.

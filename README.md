# blokus-ai

A small Blokus engine and experimentation sandbox for comparing agents, benchmarking self-play, tuning heuristics, training deep RL policies, and playing local matches through a Pygame UI.

## What is here

- `blokus_ai/core/`: board representation, move generation, rules, transforms, and game state
- `blokus_ai/agents/`: baseline, heuristic, and RL-backed agent wrappers
- `blokus_ai/rl/`: RL environment, encoders, and PyTorch actor-critic policy modules
- `blokus_ai/ui/`: ASCII rendering plus the desktop Pygame viewer
- `blokus_ai/experiments/`: self-play, tournaments, benchmarking, agent comparison, genetic tuning, RL training/evaluation, and a human match launcher
- `tests/`: unit tests covering rules, move generation, rendering, and experiments

## Setup

From the repo root:

```bash
uv sync
```

This project now depends on:

- `pygame` for local UI
- `torch` for RL training and evaluation
- `numpy` for PyTorch runtime support

If you prefer plain pip:

```bash
python3 -m pip install -e .
```

## Run tests

```bash
uv run pytest
```

## Play a human match

Launch the local Pygame human-vs-agents mode:

```bash
uv run python -m blokus_ai.experiments.human_match
```

The current scripted lineup is:

- you as player 0
- `BlockingAgent`
- `AdaptiveWeightedBlockingAgent`
- `LargestFirstAgent`

Controls in the human match:

- arrow keys move the cursor
- `TAB` and `` ` `` change piece
- `Q` and `E` rotate
- `F` flips
- `H` toggles hints
- `ENTER` places a piece or passes when no legal moves remain
- close the window to exit

## Run common experiments

Random self-play benchmark:

```bash
uv run python -m blokus_ai.experiments.benchmark --agent random --games 10
```

Agent comparison:

```bash
uv run python -m blokus_ai.experiments.agent_comparison
```

Random-agent tournament benchmark:

```bash
uv run python -m blokus_ai.experiments.random_agent_benchmark
```

Genetic tuning for adaptive blocking weights:

```bash
uv run python -m blokus_ai.experiments.genetic_tuning
```

## Deep RL workflow

The project now includes a training-oriented RL stack:

- a Blokus RL environment with legal move enumeration
- raw observation and candidate-move encoders
- a PyTorch actor-critic network
- behavior-cloning warm start from `AdaptiveWeightedBlockingAgent`
- actor-critic self-play training
- checkpoint-based evaluation against heuristic opponents

### Imitation warm start

Pretrain a policy to imitate `AdaptiveWeightedBlockingAgent`:

```bash
uv run python -m blokus_ai.experiments.rl_self_play imitate \
  --games 20 \
  --epochs 3 \
  --checkpoint checkpoints/blokus_bc.pt
```

### Actor-critic self-play training

Train from scratch or continue from the warm-started policy:

```bash
uv run python -m blokus_ai.experiments.rl_self_play train \
  --episodes 100 \
  --batch-episodes 8 \
  --workers 4 \
  --warm-start-games 20 \
  --warm-start-epochs 3 \
  --checkpoint checkpoints/blokus_ac.pt
```

Useful flags:

- `--device`: Torch device such as `cpu` or `mps`
- `--workers`: number of rollout workers for parallel self-play collection
- `--batch-episodes`: rollout batch size per optimizer step
- `--warm-start-games` and `--warm-start-epochs`: imitation warm start before RL updates

### Evaluate a trained policy

Evaluate a saved checkpoint against one heuristic opponent type occupying the other three seats:

```bash
uv run python -m blokus_ai.experiments.rl_self_play eval \
  --checkpoint checkpoints/blokus_ac.pt \
  --games 20 \
  --opponent strategic-heuristic
```

Available opponents:

- `random`
- `largest`
- `blocking`
- `weighted-blocking`
- `adaptive-weighted-blocking`
- `strategic-heuristic`

## Heuristic tuning vs RL

There are now two main optimization paths in the repo:

- `genetic_tuning.py`: tune weights of hand-designed heuristic agents through evolutionary search
- `rl_self_play.py`: learn a move policy and value function from self-play, optionally warm-started by imitation

The heuristic path is still useful when you want fast, interpretable tuning of existing ideas. The RL path is meant for learning beyond the current greedy heuristics.

## Notes

- The experiment modules are meant to be easy to edit. For example, `build_entries()` in `agent_comparison.py` controls that comparison lineup directly.
- The Pygame viewer is local desktop UI only.
- The RL trainer currently defaults to `mps` when PyTorch reports it is available, otherwise it falls back to `cpu`.
- On some newer macOS releases, PyTorch may have MPS support built but still report it unavailable. In that case the RL commands will run on CPU until the upstream PyTorch issue is resolved.
- Generated folders like `build/` and Python caches are ignored in git.

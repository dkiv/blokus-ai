# blokus-ai

A small Blokus engine and experimentation sandbox for comparing agents, benchmarking self-play, tuning heuristics, and playing local matches through a Pygame UI.

## What is here

- `blokus_ai/core/`: board representation, move generation, rules, transforms, and game state
- `blokus_ai/agents/`: baseline and heuristic agents
- `blokus_ai/ui/`: ASCII rendering plus the desktop Pygame viewer
- `blokus_ai/experiments/`: self-play, tournaments, benchmarking, agent comparison, genetic tuning, and a human match launcher
- `tests/`: unit tests covering rules, move generation, rendering, and experiments

## Setup

From the repo root:

```bash
uv sync
```

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

## Notes

- The experiment modules are meant to be easy to edit. For example, `build_entries()` in `agent_comparison.py` controls that comparison lineup directly.
- The Pygame viewer is local desktop UI only.
- Generated folders like `build/` and Python caches are ignored in git.

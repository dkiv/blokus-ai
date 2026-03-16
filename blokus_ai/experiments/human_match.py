"""Launch a local human-vs-agents Blokus pygame match."""

from __future__ import annotations

from blokus_ai import AdaptiveWeightedBlockingAgent
from blokus_ai import BlockingAgent
from blokus_ai import LargestFirstAgent
from blokus_ai import run_human_match_viewer


def main() -> None:
    agents = [
        LargestFirstAgent(),
        BlockingAgent(),
        AdaptiveWeightedBlockingAgent(),
        LargestFirstAgent(),
    ]
    run_human_match_viewer(agents, human_player=0)


if __name__ == "__main__":
    main()

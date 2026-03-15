"""Timing helpers for measuring self-play throughput."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

from blokus_ai.agents.base import Agent
from blokus_ai.agents.largest_first_agent import LargestFirstAgent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.agents.weighted_blocking_agent import WeightedBlockingAgent

from .self_play import play_game


AgentFactory = Callable[[], Agent]


@dataclass
class BenchmarkResult:
    """Aggregate timing metrics for repeated games."""

    games_played: int
    total_seconds: float
    average_seconds_per_game: float
    games_per_second: float
    total_turns: int
    total_moves: int
    turns_per_second: float
    moves_per_second: float


def benchmark_games(
    agent_factories: list[AgentFactory],
    num_games: int,
) -> BenchmarkResult:
    """Run repeated games and report end-to-end throughput."""
    if num_games <= 0:
        raise ValueError("num_games must be positive.")
    if not agent_factories:
        raise ValueError("At least one agent factory is required.")

    total_turns = 0
    total_moves = 0

    start = time.perf_counter()
    for _ in range(num_games):
        agents = [factory() for factory in agent_factories]
        result = play_game(agents=agents, print_boards=False)
        total_turns += result.turn_count
        total_moves += len(result.moves)
    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        games_played=num_games,
        total_seconds=elapsed,
        average_seconds_per_game=elapsed / num_games,
        games_per_second=num_games / elapsed if elapsed > 0 else float("inf"),
        total_turns=total_turns,
        total_moves=total_moves,
        turns_per_second=total_turns / elapsed if elapsed > 0 else float("inf"),
        moves_per_second=total_moves / elapsed if elapsed > 0 else float("inf"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Blokus self-play throughput.")
    parser.add_argument(
        "--agent",
        choices=("random", "largest", "weighted-blocking"),
        default="random",
        help="Agent type to benchmark for all players.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games to run.",
    )
    args = parser.parse_args()

    factories_by_name: dict[str, AgentFactory] = {
        "random": RandomAgent,
        "largest": LargestFirstAgent,
        "weighted-blocking": WeightedBlockingAgent,
    }
    factory = factories_by_name[args.agent]
    result = benchmark_games([factory for _ in range(4)], num_games=args.games)

    print(f"Agent: {args.agent}")
    print(f"Games: {result.games_played}")
    print(f"Total time: {result.total_seconds:.6f}s")
    print(f"Avg/game: {result.average_seconds_per_game:.6f}s")
    print(f"Games/sec: {result.games_per_second:.2f}")
    print(f"Turns/sec: {result.turns_per_second:.2f}")
    print(f"Moves/sec: {result.moves_per_second:.2f}")


if __name__ == "__main__":
    main()

"""Compare several agents over multiple tournament seeds."""

from __future__ import annotations

from statistics import mean

from blokus_ai.agents.adaptive_weighted_blocking_agent import AdaptiveWeightedBlockingAgent
from blokus_ai.agents.blocking_agent import BlockingAgent
from blokus_ai.agents.largest_first_agent import LargestFirstAgent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.agents.weighted_blocking_agent import WeightedBlockingAgent

from .tournament import TournamentResult, run_tournament

GAMES_PER_SEED = 100
SEEDS = [1, 2, 3, 4, 5]


def build_agents():
    """Edit this function to change the agent lineup for the experiment."""
    return [
        RandomAgent(),
        LargestFirstAgent(),
        BlockingAgent(),
        WeightedBlockingAgent(blocked_corner_weight=1.0 / 3.0),
        AdaptiveWeightedBlockingAgent(
            early_block_weight=0.1,
            late_block_weight=1.0,
            rank_weights=(1.0, 0.6, 0.3),
        ),
    ]


def build_labels() -> list[str]:
    return [
        "RandomAgent",
        "LargestFirstAgent",
        "BlockingAgent",
        "WeightedBlockingAgent(1/3)",
        "AdaptiveWeightedBlockingAgent",
    ]


def main() -> None:
    labels = build_labels()
    results_by_seed: list[TournamentResult] = []

    for seed in SEEDS:
        result = run_tournament(
            agents=build_agents(),
            num_games=GAMES_PER_SEED,
            seed=seed,
        )
        results_by_seed.append(result)
        print(f"Seed {seed}: average game length={result.average_game_length:.2f} turns")

    print()
    print(f"Compared {len(labels)} agents over {len(SEEDS)} seeds x {GAMES_PER_SEED} games")
    print()

    ranking = sorted(
        range(len(labels)),
        key=lambda player: (
            -mean(result.win_rates[player] for result in results_by_seed),
            -mean(result.average_score_per_player[player] for result in results_by_seed),
            player,
        ),
    )

    for place, player in enumerate(ranking, start=1):
        avg_win_rate = mean(result.win_rates[player] for result in results_by_seed)
        avg_score = mean(result.average_score_per_player[player] for result in results_by_seed)
        avg_wins = mean(result.win_counts[player] for result in results_by_seed)
        print(
            f"{place}. {labels[player]} | "
            f"avg_win_rate={avg_win_rate:.3f} | "
            f"avg_wins={avg_wins:.2f} | "
            f"avg_score={avg_score:.2f}"
        )


if __name__ == "__main__":
    main()

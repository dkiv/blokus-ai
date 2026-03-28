"""Compare several agents over multiple seeds."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from statistics import mean

from blokus_ai.agents.adaptive_weighted_blocking_agent import AdaptiveWeightedBlockingAgent
from blokus_ai.agents.base import Agent
from blokus_ai.agents.blocking_agent import BlockingAgent
from blokus_ai.agents.largest_first_agent import LargestFirstAgent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.agents.strategic_heuristic_agent import StrategicHeuristicAgent
from blokus_ai.agents.weighted_blocking_agent import WeightedBlockingAgent

from .self_play import play_game

GAMES_PER_SEED = 10
SEEDS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class AgentEntry:
    """Named agent configuration for comparison experiments."""

    label: str
    prototype: Agent


@dataclass(frozen=True)
class ComparisonResult:
    """Aggregate results for one seed across many sampled matches."""

    average_score_per_agent: dict[int, float]
    win_counts: dict[int, float]
    win_rates: dict[int, float]
    games_played: dict[int, int]
    average_game_length: float


def build_entries() -> list[AgentEntry]:
    """Edit this function to change the agent lineup for the experiment."""
    return [
        AgentEntry(
            "GA AdaptiveWeightedBlocking",
            AdaptiveWeightedBlockingAgent(
                early_block_weight=0.092,
                late_block_weight=1.780,
                rank_weights=(1.327, 0.823, 0.307),
            ),
        ),
        AgentEntry("LargestFirstAgent", LargestFirstAgent()),
        AgentEntry("BlockingAgent", BlockingAgent()),
        AgentEntry("StrategicHeuristicAgent", StrategicHeuristicAgent()),
    ]


def build_agents() -> list[Agent]:
    return [entry.prototype for entry in build_entries()]


def build_labels() -> list[str]:
    return [entry.label for entry in build_entries()]


def compare_agents(
    entries: list[AgentEntry],
    num_games: int,
    seed: int,
) -> ComparisonResult:
    """Compare any number of agents by sampling 4-player matches and aggregating results."""
    if len(entries) < 2:
        raise ValueError("At least two agents are required.")
    if num_games <= 0:
        raise ValueError("num_games must be positive.")

    rng = random.Random(seed)
    total_scores = {index: 0.0 for index in range(len(entries))}
    win_counts = {index: 0.0 for index in range(len(entries))}
    games_played = {index: 0 for index in range(len(entries))}
    total_turns = 0
    seats_per_game = min(4, len(entries))

    for _ in range(num_games):
        lineup_indices = rng.sample(range(len(entries)), k=seats_per_game)
        agents = [copy.deepcopy(entries[index].prototype) for index in lineup_indices]

        for seat, agent in enumerate(agents):
            agent_rng = getattr(agent, "rng", None)
            if isinstance(agent_rng, random.Random):
                agent_rng.seed(rng.randrange(0, 2**32) + seat)

        result = play_game(agents=agents, print_boards=False)
        total_turns += result.turn_count

        for seat, score in result.scores.items():
            entry_index = lineup_indices[seat]
            total_scores[entry_index] += score
            games_played[entry_index] += 1

        shared_win_value = 1.0 / len(result.winners)
        for seat in result.winners:
            win_counts[lineup_indices[seat]] += shared_win_value

    average_score_per_agent = {
        index: (total_scores[index] / games_played[index] if games_played[index] > 0 else 0.0)
        for index in total_scores
    }
    win_rates = {
        index: (win_counts[index] / games_played[index] if games_played[index] > 0 else 0.0)
        for index in win_counts
    }

    return ComparisonResult(
        average_score_per_agent=average_score_per_agent,
        win_counts=win_counts,
        win_rates=win_rates,
        games_played=games_played,
        average_game_length=total_turns / num_games,
    )


def main() -> None:
    entries = build_entries()
    results_by_seed: list[ComparisonResult] = []

    for seed in SEEDS:
        result = compare_agents(
            entries=entries,
            num_games=GAMES_PER_SEED,
            seed=seed,
        )
        results_by_seed.append(result)
        print(f"Seed {seed}: average game length={result.average_game_length:.2f} turns")

    print()
    print(f"Compared {len(entries)} agents over {len(SEEDS)} seeds x {GAMES_PER_SEED} sampled matches")
    print()

    ranking = sorted(
        range(len(entries)),
        key=lambda player: (
            -mean(result.win_rates[player] for result in results_by_seed),
            -mean(result.average_score_per_agent[player] for result in results_by_seed),
            player,
        ),
    )

    for place, player in enumerate(ranking, start=1):
        avg_win_rate = mean(result.win_rates[player] for result in results_by_seed)
        avg_score = mean(result.average_score_per_agent[player] for result in results_by_seed)
        avg_wins = mean(result.win_counts[player] for result in results_by_seed)
        avg_games_played = mean(result.games_played[player] for result in results_by_seed)
        print(
            f"{place}. {entries[player].label} | "
            f"avg_win_rate={avg_win_rate:.3f} | "
            f"avg_wins={avg_wins:.2f} | "
            f"avg_score={avg_score:.2f} | "
            f"avg_games_played={avg_games_played:.1f}"
        )


if __name__ == "__main__":
    main()

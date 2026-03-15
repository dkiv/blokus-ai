"""Tournament helpers for comparing agent strategies over many games."""

from __future__ import annotations

import random
from dataclasses import dataclass

from blokus_ai.agents.base import Agent

from .self_play import SelfPlayResult, play_game


@dataclass
class TournamentResult:
    """Aggregate statistics from repeated games with a fixed player ordering."""

    games_played: int
    average_score_per_player: dict[int, float]
    win_counts: dict[int, float]
    win_rates: dict[int, float]
    average_game_length: float
    results: list[SelfPlayResult]


def run_tournament(
    agents: list[Agent],
    num_games: int,
    seed: int | None = None,
) -> TournamentResult:
    """Run repeated games and compute aggregate score, win, and length statistics."""
    if num_games <= 0:
        raise ValueError("num_games must be positive.")
    if not agents:
        raise ValueError("At least one agent is required.")

    total_scores = {player: 0.0 for player in range(len(agents))}
    win_counts = {player: 0.0 for player in range(len(agents))}
    total_turns = 0
    results: list[SelfPlayResult] = []

    seed_rng = random.Random(seed)

    for _ in range(num_games):
        # Re-seed random baseline agents between games while leaving deterministic agents untouched.
        for player, agent in enumerate(agents):
            agent_rng = getattr(agent, "rng", None)
            if isinstance(agent_rng, random.Random):
                agent_rng.seed(seed_rng.randrange(0, 2**32) + player)

        result = play_game(agents=agents, print_boards=False)
        results.append(result)
        total_turns += result.turn_count

        for player, score in result.scores.items():
            total_scores[player] += score

        shared_win_value = 1.0 / len(result.winners)
        for player in result.winners:
            win_counts[player] += shared_win_value

    average_score_per_player = {
        player: total_scores[player] / num_games for player in total_scores
    }
    win_rates = {player: win_counts[player] / num_games for player in win_counts}

    return TournamentResult(
        games_played=num_games,
        average_score_per_player=average_score_per_player,
        win_counts=win_counts,
        win_rates=win_rates,
        average_game_length=total_turns / num_games,
        results=results,
    )

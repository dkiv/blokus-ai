from blokus_ai.agents.largest_first_agent import LargestFirstAgent
from blokus_ai.experiments.benchmark import benchmark_games


def test_benchmark_games_returns_consistent_totals() -> None:
    result = benchmark_games([LargestFirstAgent for _ in range(4)], num_games=1)

    assert result.games_played == 1
    assert result.total_seconds >= 0
    assert result.average_seconds_per_game >= 0
    assert result.games_per_second >= 0
    assert result.total_turns >= result.total_moves
    assert result.turns_per_second >= 0
    assert result.moves_per_second >= 0

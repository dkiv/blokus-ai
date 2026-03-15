"""Run repeated random-agent games and print aggregate stats."""

from __future__ import annotations

from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.experiments.tournament import run_tournament

GAME_COUNT = 100


def main() -> None:
    agents = [RandomAgent() for _ in range(4)]
    result = run_tournament(agents=agents, num_games=GAME_COUNT, seed=17)

    print(f"Random-agent benchmark over {GAME_COUNT} games")
    print("Average score per player:")
    for player in sorted(result.average_score_per_player):
        print(f"  Player {player}: {result.average_score_per_player[player]:.2f}")

    print("Win rate per player:")
    for player in sorted(result.win_rates):
        print(f"  Player {player}: {result.win_rates[player]:.3f}")

    print(f"Average game length: {result.average_game_length:.2f} turns")


if __name__ == "__main__":
    main()

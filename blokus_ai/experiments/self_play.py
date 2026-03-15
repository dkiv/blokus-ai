"""Minimal random self-play loop for terminal inspection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.ui.ascii_renderer import render_board


@dataclass
class SelfPlayResult:
    final_state: GameState
    moves: list[Move]
    passes: int


def play_random_game(
    seed: int | None = None,
    max_turns: int | None = None,
    print_boards: bool = True,
) -> SelfPlayResult:
    """Play a random game by repeatedly sampling from legal moves."""
    state = GameState.new_game()
    agent = RandomAgent()
    if seed is not None:
        agent.rng.seed(seed)

    moves: list[Move] = []
    consecutive_passes = 0
    total_passes = 0
    turn_count = 0
    player_count = len(state.remaining_pieces)

    if print_boards:
        _print_turn_banner(state.current_player, "Initial board")
        print(render_board(state))

    while consecutive_passes < player_count:
        if max_turns is not None and turn_count >= max_turns:
            break

        player = state.current_player
        move = agent.select_move(state)

        if move is not None:
            state = state.apply_move(move)
            moves.append(move)
            consecutive_passes = 0
            turn_count += 1

            if print_boards:
                _print_turn_banner(player, f"Turn {turn_count}: {move.piece_name} at {move.origin}")
                print(render_board(state))
            continue

        total_passes += 1
        consecutive_passes += 1

        if print_boards:
            _print_turn_banner(player, "Pass")
            print(render_board(state))

        state = state.pass_turn()

    return SelfPlayResult(final_state=state, moves=moves, passes=total_passes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random Blokus self-play in the terminal.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible play.")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum number of played moves before stopping.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable board printing and only run the simulation.",
    )
    args = parser.parse_args()

    result = play_random_game(
        seed=args.seed,
        max_turns=args.max_turns,
        print_boards=not args.no_render,
    )

    if not args.no_render:
        print(f"Finished after {len(result.moves)} moves and {result.passes} passes.")


def _print_turn_banner(player: int, message: str) -> None:
    print(f"\nPlayer {player}: {message}")


if __name__ == "__main__":
    main()

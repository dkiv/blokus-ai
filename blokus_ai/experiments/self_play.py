"""Self-play runners and terminal experiment entry points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from blokus_ai.agents.base import Agent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import generate_legal_moves


@dataclass
class SelfPlayResult:
    final_state: GameState
    moves: list[Move]
    move_history: list[Move | None]
    passes: int
    scores: dict[int, int]
    winners: list[int]
    turn_count: int


@dataclass
class SelfPlaySession:
    """Stepwise self-play session for live play or replay."""

    state: GameState
    moves: list[Move]
    move_history: list[Move | None]
    passes: int
    consecutive_passes: int
    turn_count: int
    agents: list[Agent] | None = None
    replay_moves: list[Move | None] | None = None
    replay_index: int = 0

    @classmethod
    def from_agents(
        cls,
        agents: list[Agent],
        initial_state: GameState | None = None,
    ) -> SelfPlaySession:
        state = GameState.new_game(player_count=len(agents)) if initial_state is None else initial_state

        if len(agents) != len(state.remaining_pieces):
            raise ValueError("Number of agents must match the number of players in the game state.")

        return cls(
            state=state,
            moves=[],
            move_history=[],
            passes=0,
            consecutive_passes=0,
            turn_count=0,
            agents=agents,
        )

    @classmethod
    def from_move_list(
        cls,
        replay_moves: list[Move | None],
        initial_state: GameState | None = None,
        player_count: int = 4,
    ) -> SelfPlaySession:
        state = GameState.new_game(player_count=player_count) if initial_state is None else initial_state
        return cls(
            state=state,
            moves=[],
            move_history=[],
            passes=0,
            consecutive_passes=0,
            turn_count=0,
            replay_moves=list(replay_moves),
        )

    def is_finished(self) -> bool:
        if self.replay_moves is not None:
            return self.replay_index >= len(self.replay_moves)
        return self.consecutive_passes >= len(self.state.remaining_pieces)

    def result(self) -> SelfPlayResult:
        scores = self.state.scores()
        max_score = max(scores.values()) if scores else 0
        winners = [player for player, score in scores.items() if score == max_score]

        return SelfPlayResult(
            final_state=self.state,
            moves=list(self.moves),
            move_history=list(self.move_history),
            passes=self.passes,
            scores=scores,
            winners=winners,
            turn_count=self.turn_count,
        )

    def step(self) -> Move | None:
        if self.is_finished():
            return None

        if self.replay_moves is not None:
            move = self.replay_moves[self.replay_index]
            if move is None:
                self.passes += 1
                self.consecutive_passes += 1
                self.move_history.append(None)
                self.state = self.state.pass_turn()
            else:
                self.state = self.state.apply_move(move)
                self.moves.append(move)
                self.move_history.append(move)
                self.consecutive_passes = 0
            self.replay_index += 1
            self.turn_count += 1
            return move

        if self.agents is None:
            raise ValueError("Agent-based session requires agents.")

        player = self.state.current_player
        legal_moves = generate_legal_moves(self.state, player=player)

        if legal_moves:
            move = self.agents[player].select_move(self.state, legal_moves)
            if move is None:
                raise ValueError(f"Agent for player {player} cannot pass when legal moves exist.")
            if move not in legal_moves:
                raise ValueError(f"Agent for player {player} returned a move that is not legal.")
            self.state = self.state.apply_move(move)
            self.moves.append(move)
            self.move_history.append(move)
            self.consecutive_passes = 0
            self.turn_count += 1
            return move

        self.passes += 1
        self.consecutive_passes += 1
        self.move_history.append(None)
        self.turn_count += 1
        self.state = self.state.pass_turn()
        return None


def play_game(
    agents: list[Agent],
    initial_state: GameState | None = None,
    print_boards: bool = False,
    renderer: Callable[[GameState], str] | None = None,
    max_turns: int | None = None,
    debug: bool = False,
    debug_interval: int = 10,
) -> SelfPlayResult:
    """Run a full game using one agent per player."""
    session = SelfPlaySession.from_agents(agents=agents, initial_state=initial_state)

    if debug:
        print(f"[debug] starting self-play with {len(agents)} players")

    if print_boards and renderer is not None:
        _print_turn_banner(session.state.current_player, "Initial board")
        print(renderer(session.state))

    while not session.is_finished():
        if max_turns is not None and session.turn_count >= max_turns:
            break

        player = session.state.current_player
        move = session.step()

        if debug and (
            session.turn_count <= 5
            or session.turn_count % max(debug_interval, 1) == 0
            or move is None
        ):
            if move is None:
                print(
                    f"[debug] turn={session.turn_count} player={player} action=pass "
                    f"consecutive_passes={session.consecutive_passes}"
                )
            else:
                print(
                    f"[debug] turn={session.turn_count} player={player} action=move "
                    f"piece={move.piece_name} origin={move.origin}"
                )

        if print_boards and renderer is not None:
            if move is None:
                _print_turn_banner(player, "Pass")
            else:
                _print_turn_banner(
                    player,
                    f"Turn {session.turn_count}: {move.piece_name} at {move.origin}",
                )
            print(renderer(session.state))

    result = session.result()
    if debug:
        print(
            f"[debug] finished turns={result.turn_count} moves={len(result.moves)} "
            f"passes={result.passes} winners={result.winners}"
        )
    return result


def play_random_game(
    seed: int | None = None,
    print_boards: bool = False,
    renderer: Callable[[GameState], str] | None = None,
    max_turns: int | None = None,
    debug: bool = False,
    debug_interval: int = 10,
) -> SelfPlayResult:
    """Run a full random-agent self-play game."""
    agents = [RandomAgent() for _ in range(4)]
    for player, agent in enumerate(agents):
        if seed is not None:
            agent.rng.seed(seed + player)
    return play_game(
        agents=agents,
        print_boards=print_boards,
        renderer=renderer,
        max_turns=max_turns,
        debug=debug,
        debug_interval=debug_interval,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random-agent Blokus self-play in the terminal.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible play.")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional cap on the number of played moves.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable board printing and only run the simulation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print lightweight progress logs while the game runs.",
    )
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=10,
        help="When debug logging is enabled, print progress every N turns after the opening turns.",
    )
    args = parser.parse_args()

    result = play_random_game(
        seed=args.seed,
        print_boards=False,
        max_turns=args.max_turns,
        debug=args.debug,
        debug_interval=args.debug_interval,
    )

    print(f"Finished after {result.turn_count} turns, {len(result.moves)} moves, and {result.passes} passes.")
    print(f"Scores: {result.scores}")
    print(f"Winners: {result.winners}")


def _print_turn_banner(player: int, message: str) -> None:
    print(f"\nPlayer {player}: {message}")


if __name__ == "__main__":
    main()

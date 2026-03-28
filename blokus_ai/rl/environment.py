"""Sequential RL environment wrapper around the Blokus game engine."""

from __future__ import annotations

from dataclasses import dataclass

from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import generate_legal_moves

from .encoding import RLCandidateMove, RLObservation, encode_candidate_moves, encode_observation


@dataclass(frozen=True)
class RLEnvironmentStep:
    """Result of resetting or stepping the environment."""

    observation: RLObservation
    legal_moves: tuple[Move, ...]
    candidates: tuple[RLCandidateMove, ...]
    reward: float
    done: bool
    acting_player: int
    scores: dict[int, int]


class BlokusRLEnvironment:
    """Training-oriented environment with legal-action enumeration and terminal rewards."""

    def __init__(self, player_count: int = 4) -> None:
        self.player_count = player_count
        self.state = GameState.new_game(player_count=player_count)
        self._consecutive_passes = 0

    def reset(self, initial_state: GameState | None = None) -> RLEnvironmentStep:
        """Start a fresh game and return the opening decision point."""
        self.state = GameState.new_game(player_count=self.player_count) if initial_state is None else initial_state
        self._consecutive_passes = 0
        return self._snapshot(reward=0.0, done=False)

    def step(self, move: Move | None) -> RLEnvironmentStep:
        """Apply one move for the current player and advance the environment."""
        if self.is_done():
            raise ValueError("Cannot step a finished environment.")

        legal_moves = generate_legal_moves(self.state, player=self.state.current_player)
        if legal_moves:
            if move is None:
                raise ValueError("A legal move is available, so passing is not allowed.")
            if move not in legal_moves:
                raise ValueError("Move is not legal in the current state.")
            self.state = self.state.apply_move(move)
            self._consecutive_passes = 0
        else:
            if move is not None:
                raise ValueError("No legal moves are available, so the action must be a pass.")
            self.state = self.state.pass_turn()
            self._consecutive_passes += 1

        done = self.is_done()
        reward = self._terminal_reward() if done else 0.0
        return self._snapshot(reward=reward, done=done)

    def is_done(self) -> bool:
        """Return whether the current episode has ended."""
        return self._consecutive_passes >= len(self.state.remaining_pieces)

    def legal_moves(self) -> list[Move]:
        """Return legal moves for the current player."""
        return generate_legal_moves(self.state, player=self.state.current_player)

    def _snapshot(self, reward: float, done: bool) -> RLEnvironmentStep:
        legal_moves = self.legal_moves()
        return RLEnvironmentStep(
            observation=encode_observation(self.state),
            legal_moves=tuple(legal_moves),
            candidates=tuple(encode_candidate_moves(self.state, legal_moves)),
            reward=reward,
            done=done,
            acting_player=self.state.current_player,
            scores=self.state.scores(),
        )

    def _terminal_reward(self) -> float:
        """Normalize the final margin for the acting player in [-1, 1]."""
        scores = self.state.scores()
        current_score = scores[self.state.current_player]
        best_opponent_score = max(
            score for player, score in scores.items() if player != self.state.current_player
        )
        return (current_score - best_opponent_score) / 89.0

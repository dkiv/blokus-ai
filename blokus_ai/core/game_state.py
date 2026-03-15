"""Overall game state and turn management."""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Board
from .move import Move
from .pieces import ALL_PIECES


@dataclass
class GameState:
    """Current board position and piece inventory for each player."""

    board: Board = field(default_factory=Board)
    current_player: int = 0
    remaining_pieces: dict[int, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.remaining_pieces:
            self.remaining_pieces = {player: ALL_PIECES for player in range(4)}

    @classmethod
    def new_game(cls, player_count: int = 4) -> GameState:
        if player_count <= 0:
            raise ValueError("player_count must be positive.")
        return cls(
            board=Board(),
            current_player=0,
            remaining_pieces={player: ALL_PIECES for player in range(player_count)},
        )

    def apply_move(self, move: Move) -> GameState:
        from .rules import validate_move

        if move.player != self.current_player:
            raise ValueError("It is not this player's turn.")

        validate_move(self, move, player=move.player)

        available = self.remaining_pieces.get(move.player, frozenset())
        next_board = self.board.clone()
        next_board.place(move.placed_cells, player=move.player)

        next_remaining = dict(self.remaining_pieces)
        next_remaining[move.player] = frozenset(
            piece for piece in available if piece != move.piece_name
        )

        players = sorted(next_remaining)
        next_index = (players.index(self.current_player) + 1) % len(players)

        return GameState(
            board=next_board,
            current_player=players[next_index],
            remaining_pieces=next_remaining,
        )

    def pass_turn(self) -> GameState:
        players = sorted(self.remaining_pieces)
        next_index = (players.index(self.current_player) + 1) % len(players)

        return GameState(
            board=self.board,
            current_player=players[next_index],
            remaining_pieces=dict(self.remaining_pieces),
        )

    def scores(self) -> dict[int, int]:
        return {
            player: self.board.player_counts.get(player, 0) for player in self.remaining_pieces
        }

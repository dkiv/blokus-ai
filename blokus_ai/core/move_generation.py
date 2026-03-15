"""Simple legal move generation for Blokus."""

from __future__ import annotations

from .game_state import GameState
from .move import Move
from .pieces import PIECE_TRANSFORMS
from .rules import is_legal_move


def generate_legal_moves(state: GameState, player: int | None = None) -> list[Move]:
    """Return all legal moves for the target player."""
    active_player = state.current_player if player is None else player

    if active_player not in state.remaining_pieces:
        return []

    legal_moves: list[Move] = []

    for piece_name in sorted(state.remaining_pieces[active_player]):
        for cells in PIECE_TRANSFORMS[piece_name]:
            for row in range(state.board.size):
                for col in range(state.board.size):
                    move = Move(
                        player=active_player,
                        piece_name=piece_name,
                        origin=(row, col),
                        cells=cells,
                    )
                    if is_legal_move(state, move, player=active_player):
                        legal_moves.append(move)

    return legal_moves

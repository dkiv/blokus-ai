"""ASCII rendering helpers for terminal inspection."""

from __future__ import annotations

from blokus_ai.core.game_state import GameState

PLAYER_SYMBOLS: dict[int, str] = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}
EMPTY_SYMBOL = "."


def render_board(state: GameState) -> str:
    """Render the current board as ASCII text."""
    header = "   " + " ".join(f"{col:02d}" for col in range(state.board.size))
    rows = [header]

    for row_index, row in enumerate(state.board.grid):
        rendered_row = " ".join(PLAYER_SYMBOLS.get(cell, EMPTY_SYMBOL) for cell in row)
        rows.append(f"{row_index:02d} {rendered_row}")

    return "\n".join(rows)

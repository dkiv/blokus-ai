"""Random-move baseline agent."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import generate_legal_moves


@dataclass
class RandomAgent:
    """Baseline agent that samples uniformly from legal moves."""

    rng: random.Random = field(default_factory=random.Random)

    def select_move(self, state: GameState) -> Move | None:
        legal_moves = generate_legal_moves(state, player=state.current_player)
        if not legal_moves:
            return None
        return self.rng.choice(legal_moves)

"""Random-move baseline agent."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .base import Agent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move


@dataclass
class RandomAgent(Agent):
    """Baseline agent that samples uniformly from legal moves."""

    rng: random.Random = field(default_factory=random.Random)

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        if not legal_moves:
            return None
        return self.rng.choice(legal_moves)

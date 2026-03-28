"""Framework-agnostic RL agent interfaces."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from blokus_ai.agents.base import Agent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move

from .encoding import RLCandidateMove, RLObservation, encode_candidate_moves, encode_observation


class RLPolicy(Protocol):
    """Policy interface for scoring or sampling among legal candidate moves."""

    def score_actions(
        self,
        observation: RLObservation,
        candidates: Sequence[RLCandidateMove],
    ) -> Sequence[float]:
        """Return one scalar score per candidate move."""


@dataclass
class RandomRolloutPolicy:
    """Simple fallback policy used before a learned network is attached."""

    rng: random.Random = field(default_factory=random.Random)

    def score_actions(
        self,
        observation: RLObservation,
        candidates: Sequence[RLCandidateMove],
    ) -> Sequence[float]:
        return [self.rng.random() for _ in candidates]


@dataclass
class RLPolicyAgent(Agent):
    """Agent that selects among legal moves using an external RL policy object."""

    policy: RLPolicy
    sample_actions: bool = False
    temperature: float = 1.0
    rng: random.Random = field(default_factory=random.Random)

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        if not legal_moves:
            return None

        observation = encode_observation(state)
        candidates = encode_candidate_moves(state, legal_moves)
        scores = list(self.policy.score_actions(observation, candidates))

        if len(scores) != len(legal_moves):
            raise ValueError("Policy returned a score count that does not match legal moves.")

        if self.sample_actions:
            selected_index = self._sample_index(scores)
        else:
            selected_index = max(range(len(scores)), key=scores.__getitem__)
        return legal_moves[selected_index]

    def _sample_index(self, scores: Sequence[float]) -> int:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive when sampling actions.")

        scaled_scores = [score / self.temperature for score in scores]
        max_score = max(scaled_scores)
        weights = [math.exp(score - max_score) for score in scaled_scores]
        total_weight = sum(weights)
        cutoff = self.rng.random() * total_weight

        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= cutoff:
                return index
        return len(weights) - 1

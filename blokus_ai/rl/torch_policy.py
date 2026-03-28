"""PyTorch policy modules for scoring legal Blokus moves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from .agent import RLPolicy
from .encoding import RLCandidateMove, RLObservation, piece_names


def _ensure_torch_available() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for torch_policy but is not available.")


@dataclass(frozen=True)
class TorchPolicyBatch:
    """Tensor batch for one state and its current legal action set."""

    board: Tensor
    remaining: Tensor
    scores: Tensor
    turn_index: Tensor
    placed_masks: Tensor
    piece_indices: Tensor
    origins: Tensor
    cell_counts: Tensor
    extents: Tensor


class BlokusActorCriticNet(nn.Module):
    """Shared actor-critic network for variable legal move sets."""

    def __init__(
        self,
        board_channels: int = 4,
        piece_embedding_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        piece_count = len(piece_names())

        self.board_encoder = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
        )
        self.remaining_encoder = nn.Sequential(
            nn.Linear(board_channels * piece_count + board_channels + 1, hidden_dim),
            nn.ReLU(),
        )
        self.piece_embedding = nn.Embedding(piece_count, piece_embedding_dim)
        self.move_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 8 + piece_embedding_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: TorchPolicyBatch) -> tuple[Tensor, Tensor]:
        context_board = self.board_encoder(batch.board.unsqueeze(0))
        context_scalar = self.remaining_encoder(
            torch.cat(
                [
                    batch.remaining.flatten().unsqueeze(0),
                    batch.scores.unsqueeze(0),
                    batch.turn_index.unsqueeze(0),
                ],
                dim=1,
            )
        )
        state_features = torch.cat([context_board, context_scalar], dim=1)

        move_features = self.move_encoder(batch.placed_masks.unsqueeze(1))
        piece_features = self.piece_embedding(batch.piece_indices)
        action_features = torch.cat(
            [
                move_features,
                piece_features,
                batch.origins,
                batch.cell_counts.unsqueeze(1),
                batch.extents,
            ],
            dim=1,
        )
        repeated_state = state_features.expand(action_features.shape[0], -1)
        logits = self.actor_head(torch.cat([repeated_state, action_features], dim=1)).squeeze(1)
        value = self.value_head(state_features).squeeze(1)
        return logits, value


class TorchRLPolicy(RLPolicy):
    """Adapter exposing a PyTorch actor-critic network through the RLPolicy interface."""

    def __init__(
        self,
        network: BlokusActorCriticNet | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        _ensure_torch_available()
        self.device = torch.device(device or "cpu")
        self.network = (BlokusActorCriticNet() if network is None else network).to(self.device)

    def score_actions(
        self,
        observation: RLObservation,
        candidates: Sequence[RLCandidateMove],
    ) -> Sequence[float]:
        if not candidates:
            return []
        self.network.eval()
        with torch.no_grad():
            batch = batchify_policy_inputs(observation, candidates, device=self.device)
            logits, _ = self.network(batch)
        return logits.detach().cpu().tolist()

    def save(self, path: str | Path) -> None:
        """Persist the network weights to disk."""
        torch.save(self.network.state_dict(), Path(path))

    def load(self, path: str | Path) -> None:
        """Load network weights from disk."""
        state_dict = torch.load(Path(path), map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)


def load_torch_policy(
    path: str | Path,
    device: str | torch.device | None = None,
) -> TorchRLPolicy:
    """Construct a policy and hydrate it from a checkpoint."""
    policy = TorchRLPolicy(device=device)
    policy.load(path)
    return policy


def resolve_torch_device(preferred_device: str | None = None) -> torch.device:
    """Pick a sensible default device, preferring Apple MPS when available."""
    if preferred_device:
        return torch.device(preferred_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batchify_policy_inputs(
    observation: RLObservation,
    candidates: Sequence[RLCandidateMove],
    device: str | torch.device = "cpu",
) -> TorchPolicyBatch:
    """Convert one observation and its legal candidates into tensors."""
    piece_index_lookup = {piece_name: index for index, piece_name in enumerate(piece_names())}

    return TorchPolicyBatch(
        board=torch.tensor(observation.board_planes, dtype=torch.float32, device=device),
        remaining=torch.tensor(observation.remaining_piece_planes, dtype=torch.float32, device=device),
        scores=torch.tensor(observation.scores, dtype=torch.float32, device=device) / 89.0,
        turn_index=torch.tensor([observation.turn_index / 89.0], dtype=torch.float32, device=device),
        placed_masks=torch.tensor(
            [candidate.placed_mask for candidate in candidates],
            dtype=torch.float32,
            device=device,
        ),
        piece_indices=torch.tensor(
            [piece_index_lookup[candidate.move.piece_name] for candidate in candidates],
            dtype=torch.long,
            device=device,
        ),
        origins=torch.tensor([candidate.origin for candidate in candidates], dtype=torch.float32, device=device),
        cell_counts=torch.tensor(
            [candidate.cell_count / 5.0 for candidate in candidates],
            dtype=torch.float32,
            device=device,
        ),
        extents=torch.tensor(
            [
                (
                    candidate.transform_extent[0] / 5.0,
                    candidate.transform_extent[1] / 5.0,
                )
                for candidate in candidates
            ],
            dtype=torch.float32,
            device=device,
        ),
    )

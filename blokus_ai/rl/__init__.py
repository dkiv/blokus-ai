"""RL building blocks for training Blokus agents from self-play."""

from .agent import RLPolicy, RLPolicyAgent, RandomRolloutPolicy
from .encoding import (
    RLCandidateMove,
    RLObservation,
    encode_candidate_move,
    encode_candidate_moves,
    encode_observation,
    piece_names,
)
from .environment import BlokusRLEnvironment, RLEnvironmentStep

try:
    from .torch_policy import (
        BlokusActorCriticNet,
        TorchPolicyBatch,
        TorchRLPolicy,
        batchify_policy_inputs,
        load_torch_policy,
        resolve_torch_device,
    )
except ModuleNotFoundError:
    BlokusActorCriticNet = None
    TorchPolicyBatch = None
    TorchRLPolicy = None
    batchify_policy_inputs = None
    load_torch_policy = None
    resolve_torch_device = None

__all__ = [
    "BlokusRLEnvironment",
    "RLCandidateMove",
    "RLEnvironmentStep",
    "RLObservation",
    "RLPolicy",
    "RLPolicyAgent",
    "RandomRolloutPolicy",
    "encode_candidate_move",
    "encode_candidate_moves",
    "encode_observation",
    "piece_names",
]

if TorchRLPolicy is not None:
    __all__.extend(
        [
            "BlokusActorCriticNet",
            "TorchPolicyBatch",
            "TorchRLPolicy",
            "batchify_policy_inputs",
            "load_torch_policy",
            "resolve_torch_device",
        ]
    )

"""Top-level exports for the Blokus engine package."""

from importlib import import_module

from .agents import (
    AdaptiveWeightedBlockingAgent,
    Agent,
    BlockingAgent,
    LargestFirstAgent,
    RandomAgent,
    RLPolicyAgent,
    StrategicHeuristicAgent,
    WeightedBlockingAgent,
)
from .core import (
    ALL_PIECES,
    BOARD_SIZE,
    Board,
    count_legal_moves,
    Coordinate,
    frontier_targets,
    GameState,
    Move,
    PIECES,
    PIECE_TRANSFORMS,
    STARTING_CORNERS,
    Shape,
    generate_legal_moves,
    is_legal_move,
    validate_move,
)
from .rl import (
    BlokusActorCriticNet,
    BlokusRLEnvironment,
    RLCandidateMove,
    RLEnvironmentStep,
    RLObservation,
    RLPolicy,
    RandomRolloutPolicy,
    TorchPolicyBatch,
    TorchRLPolicy,
    batchify_policy_inputs,
    encode_candidate_move,
    encode_candidate_moves,
    encode_observation,
    load_torch_policy,
    piece_names,
    resolve_torch_device,
)
from .ui import render_board


def run_move_replay_viewer(*args, **kwargs):
    from .ui import run_move_replay_viewer as _run_move_replay_viewer

    return _run_move_replay_viewer(*args, **kwargs)


def run_agent_match_viewer(*args, **kwargs):
    from .ui import run_agent_match_viewer as _run_agent_match_viewer

    return _run_agent_match_viewer(*args, **kwargs)


def run_human_match_viewer(*args, **kwargs):
    from .ui import run_human_match_viewer as _run_human_match_viewer

    return _run_human_match_viewer(*args, **kwargs)


def run_random_self_play_viewer(*args, **kwargs):
    from .ui import run_random_self_play_viewer as _run_random_self_play_viewer

    return _run_random_self_play_viewer(*args, **kwargs)


_EXPERIMENT_EXPORTS = {
    "BenchmarkResult",
    "ImitationStats",
    "RLEvalStats",
    "RLSelfPlayStats",
    "SelfPlayResult",
    "SelfPlaySession",
    "TournamentResult",
    "benchmark_games",
    "evaluate_policy",
    "play_game",
    "play_random_game",
    "run_tournament",
    "train_actor_critic",
    "train_policy_gradient",
    "warm_start_policy",
}


def __getattr__(name: str):
    if name not in _EXPERIMENT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    experiments = import_module(".experiments", __name__)
    value = getattr(experiments, name)
    globals()[name] = value
    return value

__all__ = [
    "ALL_PIECES",
    "Agent",
    "AdaptiveWeightedBlockingAgent",
    "BlockingAgent",
    "BOARD_SIZE",
    "BenchmarkResult",
    "Board",
    "count_legal_moves",
    "Coordinate",
    "GameState",
    "LargestFirstAgent",
    "Move",
    "PIECE_TRANSFORMS",
    "PIECES",
    "RandomAgent",
    "RandomRolloutPolicy",
    "ImitationStats",
    "RLEvalStats",
    "RLSelfPlayStats",
    "RLCandidateMove",
    "RLEnvironmentStep",
    "RLObservation",
    "RLPolicy",
    "RLPolicyAgent",
    "STARTING_CORNERS",
    "StrategicHeuristicAgent",
    "SelfPlayResult",
    "SelfPlaySession",
    "Shape",
    "TournamentResult",
    "WeightedBlockingAgent",
    "generate_legal_moves",
    "frontier_targets",
    "is_legal_move",
    "benchmark_games",
    "evaluate_policy",
    "BlokusRLEnvironment",
    "BlokusActorCriticNet",
    "TorchPolicyBatch",
    "TorchRLPolicy",
    "batchify_policy_inputs",
    "encode_candidate_move",
    "encode_candidate_moves",
    "encode_observation",
    "load_torch_policy",
    "play_game",
    "play_random_game",
    "piece_names",
    "render_board",
    "resolve_torch_device",
    "run_tournament",
    "train_actor_critic",
    "train_policy_gradient",
    "warm_start_policy",
    "run_move_replay_viewer",
    "run_agent_match_viewer",
    "run_human_match_viewer",
    "run_random_self_play_viewer",
    "validate_move",
]

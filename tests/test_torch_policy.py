import pytest

pytest.importorskip("torch")

from blokus_ai.core.game_state import GameState
from blokus_ai.rl import (
    BlokusActorCriticNet,
    TorchRLPolicy,
    batchify_policy_inputs,
    encode_candidate_moves,
    encode_observation,
    load_torch_policy,
)


def test_torch_policy_batch_shapes_match_legal_action_count() -> None:
    state = GameState.new_game()
    observation = encode_observation(state)
    candidates = encode_candidate_moves(state)
    batch = batchify_policy_inputs(observation, candidates)

    assert batch.board.shape == (4, 20, 20)
    assert batch.remaining.shape[0] == 4
    assert batch.placed_masks.shape[0] == len(candidates)
    assert batch.piece_indices.shape[0] == len(candidates)


def test_torch_actor_critic_net_emits_one_logit_per_candidate_and_value() -> None:
    state = GameState.new_game()
    observation = encode_observation(state)
    candidates = encode_candidate_moves(state)
    batch = batchify_policy_inputs(observation, candidates)

    network = BlokusActorCriticNet()
    logits, value = network(batch)

    assert logits.shape == (len(candidates),)
    assert value.shape == (1,)


def test_torch_policy_can_round_trip_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "policy.pt"
    policy = TorchRLPolicy()
    policy.save(checkpoint_path)

    loaded_policy = load_torch_policy(checkpoint_path)
    logits, value = loaded_policy.network(policy_batch := batchify_policy_inputs(
        encode_observation(GameState.new_game()),
        encode_candidate_moves(GameState.new_game()),
    ))

    assert checkpoint_path.exists()
    assert logits.shape[0] == policy_batch.placed_masks.shape[0]
    assert value.shape == (1,)

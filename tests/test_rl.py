from dataclasses import dataclass

from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.pieces import PIECES
from blokus_ai.rl import BlokusRLEnvironment, RLPolicyAgent, piece_names


def test_rl_environment_reset_exposes_opening_legal_moves() -> None:
    environment = BlokusRLEnvironment()

    step = environment.reset()

    assert not step.done
    assert step.reward == 0.0
    assert step.acting_player == 0
    assert step.legal_moves
    assert len(step.legal_moves) == len(step.candidates)
    assert len(step.observation.board_planes) == 4
    assert len(step.observation.remaining_piece_planes[0]) == len(piece_names())


def test_rl_environment_rejects_pass_when_a_legal_move_exists() -> None:
    environment = BlokusRLEnvironment()
    environment.reset()

    try:
        environment.step(None)
    except ValueError as exc:
        assert "passing is not allowed" in str(exc)
    else:
        raise AssertionError("Expected a ValueError when passing with legal moves available.")


def test_rl_environment_returns_terminal_reward_after_final_pass() -> None:
    environment = BlokusRLEnvironment(player_count=2)
    initial_state = GameState(
        current_player=0,
        remaining_pieces={0: frozenset(), 1: frozenset()},
    )
    initial_state.board.place({(0, 0), (0, 1)}, player=0)
    initial_state.board.place({(19, 19)}, player=1)

    opening = environment.reset(initial_state=initial_state)
    assert not opening.legal_moves

    first_pass = environment.step(None)
    assert not first_pass.done
    assert first_pass.reward == 0.0

    second_pass = environment.step(None)
    assert second_pass.done
    assert second_pass.reward > 0.0


@dataclass
class LargestCellPolicy:
    def score_actions(self, observation, candidates):
        return [candidate.cell_count for candidate in candidates]


def test_rl_policy_agent_can_pick_moves_from_candidate_encodings() -> None:
    agent = RLPolicyAgent(policy=LargestCellPolicy())
    state = GameState.new_game()
    legal_moves = [
        Move(player=0, piece_name="I1", origin=(0, 0), cells=PIECES["I1"]),
        Move(player=0, piece_name="I2", origin=(0, 0), cells=PIECES["I2"]),
    ]

    assert agent.select_move(state, legal_moves) == legal_moves[1]

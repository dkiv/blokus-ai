from dataclasses import dataclass

import pytest

from blokus_ai.agents.base import Agent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.pieces import PIECES
from blokus_ai.experiments.self_play import SelfPlayResult, play_game
from blokus_ai.experiments.tournament import run_tournament


@dataclass
class FirstMoveAgent(Agent):
    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        return legal_moves[0]


@dataclass
class IllegalMoveAgent(Agent):
    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        return Move(player=state.current_player, piece_name="I1", origin=(5, 5), cells=PIECES["I1"])


@dataclass
class PassingAgent(Agent):
    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move | None:
        return None


def test_play_game_returns_scores_for_all_players() -> None:
    result = play_game(
        agents=[FirstMoveAgent() for _ in range(4)],
        print_boards=False,
        max_turns=4,
    )

    assert set(result.scores) == {0, 1, 2, 3}
    assert sum(result.scores.values()) == sum(len(move.cells) for move in result.moves)
    assert result.passes == 0
    assert result.turn_count == len(result.move_history)
    assert result.winners


def test_play_game_rejects_illegal_agent_choice() -> None:
    agents: list[Agent] = [IllegalMoveAgent(), FirstMoveAgent(), FirstMoveAgent(), FirstMoveAgent()]

    with pytest.raises(ValueError, match="not legal"):
        play_game(agents=agents, print_boards=False)


def test_play_game_rejects_passing_when_legal_moves_exist() -> None:
    agents: list[Agent] = [PassingAgent(), FirstMoveAgent(), FirstMoveAgent(), FirstMoveAgent()]

    with pytest.raises(ValueError, match="cannot pass"):
        play_game(agents=agents, print_boards=False)


def test_tournament_statistics_aggregate_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted_results = [
        SelfPlayResult(
            final_state=GameState.new_game(),
            moves=[],
            move_history=[None, None],
            passes=2,
            scores={0: 5, 1: 3, 2: 3, 3: 1},
            winners=[0],
            turn_count=2,
        ),
        SelfPlayResult(
            final_state=GameState.new_game(),
            moves=[],
            move_history=[None, None, None],
            passes=3,
            scores={0: 2, 1: 7, 2: 7, 3: 1},
            winners=[1, 2],
            turn_count=3,
        ),
    ]
    result_iter = iter(scripted_results)

    def fake_play_game(*, agents: list[Agent], print_boards: bool = False):
        return next(result_iter)

    monkeypatch.setattr("blokus_ai.experiments.tournament.play_game", fake_play_game)

    result = run_tournament(
        agents=[RandomAgent(), RandomAgent(), RandomAgent(), RandomAgent()],
        num_games=2,
        seed=11,
    )

    assert result.games_played == 2
    assert set(result.average_score_per_player) == {0, 1, 2, 3}
    assert set(result.win_counts) == {0, 1, 2, 3}
    assert set(result.win_rates) == {0, 1, 2, 3}
    assert len(result.results) == 2
    assert sum(result.win_counts.values()) == pytest.approx(2.0)
    assert sum(result.win_rates.values()) == pytest.approx(1.0)
    assert result.average_game_length > 0

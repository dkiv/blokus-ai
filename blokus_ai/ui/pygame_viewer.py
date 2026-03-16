"""Minimal pygame viewer for Blokus self-play, replay, and human play."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Callable

from blokus_ai.agents.adaptive_weighted_blocking_agent import AdaptiveWeightedBlockingAgent
from blokus_ai.agents.base import Agent
from blokus_ai.agents.blocking_agent import BlockingAgent
from blokus_ai.agents.largest_first_agent import LargestFirstAgent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.agents.weighted_blocking_agent import WeightedBlockingAgent
from blokus_ai.core.board import BOARD_SIZE
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import generate_legal_moves
from blokus_ai.core.pieces import PIECES
from blokus_ai.core.rules import is_legal_move
from blokus_ai.core.transforms import reflect_horizontal, rotate_clockwise
from blokus_ai.experiments.self_play import SelfPlayResult, SelfPlaySession, play_game, play_random_game

PLAYER_COLORS: dict[int, tuple[int, int, int]] = {
    0: (220, 75, 65),
    1: (70, 130, 220),
    2: (70, 170, 95),
    3: (230, 180, 60),
}
BACKGROUND_COLOR = (242, 240, 233)
GRID_COLOR = (155, 150, 140)
EMPTY_CELL_COLOR = (250, 248, 242)
TEXT_COLOR = (40, 40, 40)
PANEL_COLOR = (228, 224, 214)
ACCENT_COLOR = (115, 96, 60)
HIGHLIGHT_COLOR = (255, 238, 173)
INVALID_PREVIEW_COLOR = (240, 180, 180)


@dataclass
class ViewerConfig:
    cell_size: int = 28
    margin: int = 24
    sidebar_width: int = 220
    footer_height: int = 420
    autoplay_delay_ms: int = 250
    window_title: str = "Blokus Viewer"


class PygameViewer:
    """Simple viewer that can step through live self-play or a replay move list."""

    def __init__(
        self,
        session_factory: Callable[[], SelfPlaySession],
        config: ViewerConfig | None = None,
        player_labels: list[str] | None = None,
        final_result: SelfPlayResult | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.config = ViewerConfig() if config is None else config
        self.player_labels = player_labels
        self.final_result = final_result
        self.session = self._session_factory()
        self.autoplay = False

    def reset(self) -> None:
        self.session = self._session_factory()
        self.autoplay = False

    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "pygame is required for the viewer. Install it with `pip install pygame`."
            ) from exc

        pygame.init()
        pygame.font.init()

        board_pixels = BOARD_SIZE * self.config.cell_size
        width = board_pixels + self.config.sidebar_width + self.config.margin * 3
        height = board_pixels + self.config.margin * 3 + self.config.footer_height

        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(self.config.window_title)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 28)
        small_font = pygame.font.SysFont(None, 22)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self._advance_one_step()
                    elif event.key == pygame.K_p:
                        self.autoplay = not self.autoplay
                    elif event.key == pygame.K_r:
                        self.reset()

            if self.autoplay and not self.session.is_finished():
                self._advance_one_step()
                pygame.time.delay(self.config.autoplay_delay_ms)

            self._draw(screen, font, small_font, pygame)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        if self.final_result is not None:
            _print_final_standings(self.final_result, self.player_labels)

    def _advance_one_step(self) -> None:
        if not self.session.is_finished():
            self.session.step()

    def _draw(self, screen, font, small_font, pygame) -> None:
        screen.fill(BACKGROUND_COLOR)

        board_left = self.config.margin
        board_top = self.config.margin
        board_size_pixels = BOARD_SIZE * self.config.cell_size

        self._draw_board(screen, board_left, board_top, pygame)

        panel_left = board_left + board_size_pixels + self.config.margin
        panel_rect = pygame.Rect(
            panel_left,
            board_top,
            self.config.sidebar_width,
            board_size_pixels,
        )
        pygame.draw.rect(screen, PANEL_COLOR, panel_rect, border_radius=12)

        y = board_top + 18
        info_lines = self._info_lines()
        for index, text in enumerate(info_lines):
            current_font = font if index < 2 else small_font
            surface = current_font.render(text, True, TEXT_COLOR)
            screen.blit(surface, (panel_left + 16, y))
            y += 34 if index < 2 else 26

        footer_top = board_top + board_size_pixels + self.config.margin
        footer_rect = pygame.Rect(
            board_left,
            footer_top,
            board_size_pixels + self.config.sidebar_width + self.config.margin,
            self.config.footer_height,
        )
        pygame.draw.rect(screen, PANEL_COLOR, footer_rect, border_radius=12)
        footer_lines = [
            "Controls:",
            "SPACE step",
            "P autoplay",
            "R reset",
            "ESC quit",
        ]
        x = board_left + 16
        for text in footer_lines:
            surface = small_font.render(text, True, TEXT_COLOR)
            screen.blit(surface, (x, footer_top + 18))
            x += surface.get_width() + 18

    def _draw_board(self, screen, board_left: int, board_top: int, pygame) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                rect = pygame.Rect(
                    board_left + col * self.config.cell_size,
                    board_top + row * self.config.cell_size,
                    self.config.cell_size,
                    self.config.cell_size,
                )
                occupant = self.session.state.board.grid[row][col]
                color = EMPTY_CELL_COLOR if occupant is None else PLAYER_COLORS[occupant]
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRID_COLOR, rect, width=1)

    def _info_lines(self) -> list[str]:
        scores = self.session.state.scores()
        mode = "Replay" if self.session.replay_moves is not None else "Random Self-Play"
        status = "Finished" if self.session.is_finished() else ("Autoplay" if self.autoplay else "Paused")

        lines = [
            mode,
            f"Turn: {self.session.turn_count}",
            f"Current player: {self.session.state.current_player}",
            f"Passes: {self.session.passes}",
            f"Status: {status}",
            "Players:",
        ]
        lines.extend(self._player_line(player) for player in sorted(self.session.state.remaining_pieces))
        lines.append("Scores:")
        lines.extend(f"{self._player_label(player)}: {score}" for player, score in sorted(scores.items()))
        return lines

    def _player_label(self, player: int) -> str:
        if self.player_labels is not None and 0 <= player < len(self.player_labels):
            return self.player_labels[player]
        return f"Player {player}"

    def _player_line(self, player: int) -> str:
        return f"P{player}: {self._player_label(player)}"


class HumanVsAgentsViewer:
    """Interactive pygame mode for a human to play against AI agents."""

    def __init__(
        self,
        agents: list[Agent],
        human_player: int = 0,
        config: ViewerConfig | None = None,
    ) -> None:
        if not 0 <= human_player < len(agents):
            raise ValueError("human_player must point to a valid player.")

        self.agents = agents
        self.human_player = human_player
        self.config = ViewerConfig(window_title="Blokus Human Match") if config is None else config
        self.state = GameState.new_game(player_count=len(agents))
        self.turn_count = 0
        self.pass_count = 0
        self.consecutive_passes = 0
        self.finished = False
        self.final_result: SelfPlayResult | None = None
        self.selected_piece_index = 0
        self.selected_cells = PIECES["I1"]
        self.cursor = (0, 0)
        self.status_message = "Your move."
        self._human_legal_moves: list[Move] = []
        self.show_hints = True
        self._reset_human_selection()

    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "pygame is required for the viewer. Install it with `pip install pygame`."
            ) from exc

        pygame.init()
        pygame.font.init()

        board_pixels = BOARD_SIZE * self.config.cell_size
        width = board_pixels + self.config.sidebar_width + self.config.margin * 3
        height = board_pixels + self.config.margin * 3 + self.config.footer_height

        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(self.config.window_title)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 28)
        small_font = pygame.font.SysFont(None, 20)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._reset_game()
                    else:
                        self._handle_key(event.key, pygame)

            if not self.finished and self.state.current_player != self.human_player:
                pygame.time.delay(self.config.autoplay_delay_ms)
                self._advance_ai_turn()

            self._draw(screen, font, small_font, pygame)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        if self.final_result is not None:
            labels = [
                "You" if player == self.human_player else _agent_label(agent, player)
                for player, agent in enumerate(self.agents)
            ]
            _print_final_standings(self.final_result, labels)

    def _reset_game(self) -> None:
        self.state = GameState.new_game(player_count=len(self.agents))
        self.turn_count = 0
        self.pass_count = 0
        self.consecutive_passes = 0
        self.finished = False
        self.final_result = None
        self.status_message = "Your move."
        self._human_legal_moves = []
        self.show_hints = True
        self._reset_human_selection()

    def _handle_key(self, key, pygame) -> None:
        if self.finished or self.state.current_player != self.human_player:
            return

        if self._must_pass():
            if key in (pygame.K_RETURN, pygame.K_BACKSPACE, pygame.K_p):
                self._pass_turn("No legal moves remain. You pass.")
            return

        if key == pygame.K_LEFT:
            self.cursor = (self.cursor[0], max(0, self.cursor[1] - 1))
        elif key == pygame.K_RIGHT:
            self.cursor = (self.cursor[0], min(BOARD_SIZE - 1, self.cursor[1] + 1))
        elif key == pygame.K_UP:
            self.cursor = (max(0, self.cursor[0] - 1), self.cursor[1])
        elif key == pygame.K_DOWN:
            self.cursor = (min(BOARD_SIZE - 1, self.cursor[0] + 1), self.cursor[1])
        elif key == pygame.K_TAB:
            self._cycle_piece(1)
        elif key == pygame.K_BACKQUOTE:
            self._cycle_piece(-1)
        elif key == pygame.K_q:
            self.selected_cells = rotate_clockwise(rotate_clockwise(rotate_clockwise(self.selected_cells)))
        elif key == pygame.K_e:
            self.selected_cells = rotate_clockwise(self.selected_cells)
        elif key == pygame.K_f:
            self.selected_cells = reflect_horizontal(self.selected_cells)
        elif key == pygame.K_h:
            self.show_hints = not self.show_hints
            self.status_message = "Hints on." if self.show_hints else "Hints off."
        elif key == pygame.K_RETURN:
            self._try_place_selected_move()
        elif key in (pygame.K_BACKSPACE, pygame.K_p):
            self.status_message = "You can only pass when no legal moves remain."

    def _advance_ai_turn(self) -> None:
        legal_moves = generate_legal_moves(self.state, player=self.state.current_player)
        if legal_moves:
            move = self.agents[self.state.current_player].select_move(self.state, legal_moves)
            if move is None or move not in legal_moves:
                raise ValueError(f"Agent for player {self.state.current_player} returned an invalid move.")
            self.state = self.state.apply_move(move)
            self.turn_count += 1
            self.consecutive_passes = 0
            self.status_message = f"{_agent_label(self.agents[move.player], move.player)} played {move.piece_name}."
        else:
            self._pass_turn(f"{_agent_label(self.agents[self.state.current_player], self.state.current_player)} passes.")

        self._check_finished()
        if not self.finished and self.state.current_player == self.human_player:
            self._reset_human_selection()
            if self._must_pass():
                self.status_message = "No legal moves remain for you. Press ENTER to pass."
            else:
                self.status_message = "Your move."

    def _try_place_selected_move(self) -> None:
        piece_name = self._selected_piece_name()
        move = Move(
            player=self.human_player,
            piece_name=piece_name,
            origin=self.cursor,
            cells=self.selected_cells,
        )
        if not is_legal_move(self.state, move, player=self.human_player):
            self.status_message = "That placement is not legal."
            return

        self.state = self.state.apply_move(move)
        self.turn_count += 1
        self.consecutive_passes = 0
        self.status_message = f"You played {piece_name}."
        self._check_finished()

    def _pass_turn(self, message: str) -> None:
        self.state = self.state.pass_turn()
        self.turn_count += 1
        self.pass_count += 1
        self.consecutive_passes += 1
        self.status_message = message
        self._check_finished()

    def _check_finished(self) -> None:
        if self.consecutive_passes >= len(self.state.remaining_pieces):
            self.finished = True
            scores = self.state.scores()
            max_score = max(scores.values()) if scores else 0
            winners = [player for player, score in scores.items() if score == max_score]
            self.final_result = SelfPlayResult(
                final_state=self.state,
                moves=[],
                move_history=[],
                passes=self.pass_count,
                scores=scores,
                winners=winners,
                turn_count=self.turn_count,
            )
            self.status_message = "Game over. Close the window for final standings."

    def _draw(self, screen, font, small_font, pygame) -> None:
        screen.fill(BACKGROUND_COLOR)
        board_left = self.config.margin
        board_top = self.config.margin
        board_pixels = BOARD_SIZE * self.config.cell_size
        footer_top = board_top + board_pixels + self.config.margin

        self._draw_board(screen, board_left, board_top, pygame)
        self._draw_sidebar(screen, board_left, board_top, board_pixels, font, small_font, pygame)
        self._draw_footer(screen, board_left, footer_top, board_pixels, font, small_font, pygame)

    def _draw_board(self, screen, board_left: int, board_top: int, pygame) -> None:
        preview_move = self._current_preview_move()
        preview_cells = preview_move.placed_cells if preview_move is not None else frozenset()
        preview_legal = preview_move is not None and is_legal_move(self.state, preview_move, player=self.human_player)
        legal_outline_cells = self._legal_outline_cells() if self.show_hints else frozenset()

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                rect = pygame.Rect(
                    board_left + col * self.config.cell_size,
                    board_top + row * self.config.cell_size,
                    self.config.cell_size,
                    self.config.cell_size,
                )
                occupant = self.state.board.grid[row][col]
                color = EMPTY_CELL_COLOR if occupant is None else PLAYER_COLORS[occupant]
                if (row, col) in preview_cells:
                    color = HIGHLIGHT_COLOR if preview_legal else INVALID_PREVIEW_COLOR
                pygame.draw.rect(screen, color, rect)
                if (row, col) in legal_outline_cells and (row, col) not in preview_cells:
                    pygame.draw.rect(screen, ACCENT_COLOR, rect, width=2)
                border_width = 2 if (row, col) == self.cursor else 1
                border_color = ACCENT_COLOR if (row, col) == self.cursor else GRID_COLOR
                pygame.draw.rect(screen, border_color, rect, width=border_width)

    def _draw_sidebar(self, screen, board_left: int, board_top: int, board_pixels: int, font, small_font, pygame) -> None:
        panel_left = board_left + board_pixels + self.config.margin
        panel_rect = pygame.Rect(panel_left, board_top, self.config.sidebar_width, board_pixels)
        pygame.draw.rect(screen, PANEL_COLOR, panel_rect, border_radius=12)

        scores = self.state.scores()
        lines = [
            "Human Match",
            f"Turn: {self.turn_count}",
            f"Current: {self._player_name(self.state.current_player)}",
            f"Passes: {self.pass_count}",
            f"Status: {'Finished' if self.finished else 'Live'}",
            "Players:",
        ]
        lines.extend(f"P{player}: {self._player_name(player)}" for player in sorted(self.state.remaining_pieces))
        lines.append("Scores:")
        lines.extend(f"{self._player_name(player)}: {score}" for player, score in sorted(scores.items()))

        y = board_top + 18
        for index, text in enumerate(lines):
            current_font = font if index < 2 else small_font
            surface = current_font.render(text, True, TEXT_COLOR)
            screen.blit(surface, (panel_left + 16, y))
            y += 34 if index < 2 else 24

    def _draw_footer(self, screen, left: int, top: int, board_pixels: int, font, small_font, pygame) -> None:
        footer_width = board_pixels + self.config.sidebar_width + self.config.margin
        footer_rect = pygame.Rect(left, top, footer_width, self.config.footer_height)
        pygame.draw.rect(screen, PANEL_COLOR, footer_rect, border_radius=12)

        status_surface = font.render(self.status_message, True, TEXT_COLOR)
        screen.blit(status_surface, (left + 16, top + 12))

        controls_block_height = 72
        piece_pool_top = top + 48
        piece_pool_height = self.config.footer_height - 48 - controls_block_height - 16
        self._draw_piece_pool(
            screen,
            left + 16,
            piece_pool_top,
            footer_width - 32,
            piece_pool_height,
            small_font,
            pygame,
        )

        controls = [
            "Controls:",
            "Arrows move cursor",
            "TAB / ` change piece",
            "Q / E rotate",
            "F flip",
            "H toggle hints",
            "ENTER place or pass",
            "R reset",
            "ESC quit",
        ]
        controls_top = top + self.config.footer_height - controls_block_height + 8
        self._draw_control_line(screen, left + 16, controls_top, controls[:5], small_font)
        self._draw_control_line(screen, left + 16, controls_top + 26, controls[5:], small_font)

    def _draw_piece_pool(self, screen, left: int, top: int, width: int, height: int, font, pygame) -> None:
        available = self._playable_piece_names()
        if not available:
            return

        cols = 4
        card_width = max(130, width // cols - 8)
        card_height = 34
        for index, piece_name in enumerate(available):
            row = index // cols
            col = index % cols
            card_left = left + col * (card_width + 8)
            card_top = top + row * (card_height + 8)
            if card_top + card_height > top + height:
                break
            rect = pygame.Rect(card_left, card_top, card_width, card_height)
            is_selected = piece_name == self._selected_piece_name()
            fill = HIGHLIGHT_COLOR if is_selected else EMPTY_CELL_COLOR
            pygame.draw.rect(screen, fill, rect, border_radius=8)
            pygame.draw.rect(screen, ACCENT_COLOR if is_selected else GRID_COLOR, rect, width=2, border_radius=8)
            surface = font.render(piece_name, True, TEXT_COLOR)
            screen.blit(surface, (card_left + 10, card_top + 8))

            preview_cells = self._piece_preview_cells(piece_name)
            min_row = min(row for row, _ in preview_cells)
            max_row = max(row for row, _ in preview_cells)
            min_col = min(col for _, col in preview_cells)
            max_col = max(col for _, col in preview_cells)
            preview_width = max_col - min_col + 1
            preview_height = max_row - min_row + 1
            preview_left = card_left + rect.width - 14 - preview_width * 8
            preview_top = card_top + max(6, (rect.height - preview_height * 8) // 2)
            for cell_row, cell_col in sorted(preview_cells):
                mini_rect = pygame.Rect(
                    preview_left + (cell_col - min_col) * 8,
                    preview_top + (cell_row - min_row) * 8,
                    7,
                    7,
                )
                pygame.draw.rect(screen, PLAYER_COLORS[self.human_player], mini_rect)

    def _reset_human_selection(self) -> None:
        self._human_legal_moves = self._compute_human_legal_moves()
        available = self._playable_piece_names()
        if not available:
            self.selected_piece_index = 0
            self.selected_cells = PIECES["I1"]
            self.cursor = (0, 0)
            return

        self.selected_piece_index = min(self.selected_piece_index, len(available) - 1)
        piece_name = available[self.selected_piece_index]
        self.cursor = self._default_cursor_for_piece(piece_name)

    def _cycle_piece(self, delta: int) -> None:
        available = self._playable_piece_names()
        if not available:
            return
        self.selected_piece_index = (self.selected_piece_index + delta) % len(available)
        piece_name = available[self.selected_piece_index]
        self.cursor = self._default_cursor_for_piece(piece_name)

    def _selected_piece_name(self) -> str:
        available = self._playable_piece_names()
        if not available:
            return "I1"
        self.selected_piece_index %= len(available)
        return available[self.selected_piece_index]

    def _default_cursor_for_piece(self, piece_name: str) -> tuple[int, int]:
        legal_moves = [move for move in self._human_legal_moves if move.piece_name == piece_name]
        if legal_moves:
            self.selected_cells = legal_moves[0].cells
            return legal_moves[0].origin
        self.selected_cells = PIECES[piece_name]
        return (0, 0)

    def _current_preview_move(self) -> Move | None:
        if self.finished or self.state.current_player != self.human_player or self._must_pass():
            return None
        return Move(
            player=self.human_player,
            piece_name=self._selected_piece_name(),
            origin=self.cursor,
            cells=self.selected_cells,
        )

    def _must_pass(self) -> bool:
        return not self._human_legal_moves

    def _compute_human_legal_moves(self) -> list[Move]:
        if self.finished or self.state.current_player != self.human_player:
            return []
        return generate_legal_moves(self.state, player=self.human_player)

    def _playable_piece_names(self) -> list[str]:
        return sorted({move.piece_name for move in self._human_legal_moves})

    def _legal_outline_cells(self) -> frozenset[tuple[int, int]]:
        if self.finished or self.state.current_player != self.human_player or self._must_pass():
            return frozenset()

        selected_piece = self._selected_piece_name()
        matching_moves = [
            move
            for move in self._human_legal_moves
            if move.piece_name == selected_piece and move.cells == self.selected_cells
        ]
        if not matching_moves:
            return frozenset()

        outlined: set[tuple[int, int]] = set()
        for move in matching_moves:
            outlined.update(move.placed_cells)
        return frozenset(outlined)

    def _player_name(self, player: int) -> str:
        if player == self.human_player:
            return "You"
        return _agent_label(self.agents[player], player)

    def _piece_preview_cells(self, piece_name: str) -> frozenset[tuple[int, int]]:
        cells = PIECES[piece_name]
        max_row = max(row for row, _ in cells)
        max_col = max(col for _, col in cells)
        if max_row > max_col:
            return rotate_clockwise(cells)
        return cells

    def _draw_control_line(self, screen, left: int, top: int, texts: list[str], font) -> None:
        x = left
        for text in texts:
            surface = font.render(text, True, TEXT_COLOR)
            screen.blit(surface, (x, top))
            x += surface.get_width() + 18


def run_random_self_play_viewer(seed: int | None = None) -> None:
    """Launch the pygame viewer in live random self-play mode."""

    def session_factory() -> SelfPlaySession:
        agents = [RandomAgent() for _ in range(4)]
        for player, agent in enumerate(agents):
            if seed is not None:
                agent.rng.seed(seed + player)
        return SelfPlaySession.from_agents(agents=agents)

    PygameViewer(session_factory=session_factory).run()


def run_move_replay_viewer(
    moves: list[Move | None],
    initial_state: GameState | None = None,
    player_count: int = 4,
    player_labels: list[str] | None = None,
    final_result: SelfPlayResult | None = None,
) -> None:
    """Launch the pygame viewer in replay mode for a precomputed move list."""

    def session_factory() -> SelfPlaySession:
        return SelfPlaySession.from_move_list(
            replay_moves=moves,
            initial_state=initial_state,
            player_count=player_count,
        )

    PygameViewer(
        session_factory=session_factory,
        player_labels=player_labels,
        final_result=final_result,
    ).run()


def run_agent_match_viewer(
    agents: list[Agent],
    initial_state: GameState | None = None,
) -> None:
    """Precompute a mixed-agent game, replay it in the viewer, and print final standings."""
    result = play_game(agents=agents, initial_state=initial_state, print_boards=False)
    player_labels = [_agent_label(agent, player) for player, agent in enumerate(agents)]
    run_move_replay_viewer(
        result.move_history,
        initial_state=initial_state,
        player_count=len(agents),
        player_labels=player_labels,
        final_result=result,
    )


def run_human_match_viewer(
    agents: list[Agent],
    human_player: int = 0,
) -> None:
    """Launch an interactive pygame match with one human player and AI opponents."""
    HumanVsAgentsViewer(agents=[copy.deepcopy(agent) for agent in agents], human_player=human_player).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a minimal pygame viewer for Blokus.")
    parser.add_argument(
        "--mode",
        choices=("random", "replay"),
        default="random",
        help="Visualize live random self-play or replay a precomputed move list.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs.")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=40,
        help="Moves to precompute when launching in replay mode.",
    )
    args = parser.parse_args()

    if args.mode == "random":
        run_random_self_play_viewer(seed=args.seed)
        return

    result = play_random_game(seed=args.seed, print_boards=False, max_turns=args.max_turns)
    run_move_replay_viewer(result.move_history, final_result=result)


def _agent_label(agent: Agent, player: int) -> str:
    if isinstance(agent, AdaptiveWeightedBlockingAgent):
        return (
            f"P{player} AdaptiveWeightedBlocking("
            f"{agent.early_block_weight:g}->{agent.late_block_weight:g})"
        )
    if isinstance(agent, WeightedBlockingAgent):
        return f"P{player} WeightedBlocking({agent.blocked_corner_weight:g})"
    return f"P{player} {agent.__class__.__name__}"


def _print_final_standings(
    result: SelfPlayResult,
    player_labels: list[str] | None = None,
) -> None:
    ordered_scores = sorted(
        result.scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    print("\nFinal standings:")
    place = 1
    previous_score: int | None = None
    for index, (player, score) in enumerate(ordered_scores, start=1):
        if previous_score is not None and score < previous_score:
            place = index
        label = player_labels[player] if player_labels is not None and player < len(player_labels) else f"Player {player}"
        winner_marker = " (winner)" if player in result.winners else ""
        print(f"{place}. {label}: {score}{winner_marker}")
        previous_score = score


if __name__ == "__main__":
    main()

"""Browser-friendly pygame UI for publishing Blokus on GitHub Pages."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import traceback

from blokus_ai.agents import AdaptiveWeightedBlockingAgent
from blokus_ai.agents import Agent
from blokus_ai.agents import LargestFirstAgent
from blokus_ai.agents import RandomAgent
from blokus_ai.agents import WeightedBlockingAgent
from blokus_ai.core.board import BOARD_SIZE
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.core.move_generation import generate_legal_moves

PLAYER_COLORS: dict[int, tuple[int, int, int]] = {
    0: (214, 78, 72),
    1: (74, 122, 214),
    2: (63, 153, 92),
    3: (221, 174, 70),
}
BACKGROUND_COLOR = (246, 242, 232)
BOARD_BACKGROUND = (252, 249, 242)
GRID_COLOR = (174, 168, 155)
TEXT_COLOR = (45, 44, 40)
PANEL_COLOR = (232, 225, 210)
PANEL_BORDER = (162, 149, 123)
BUTTON_COLOR = (219, 207, 180)
BUTTON_ACTIVE_COLOR = (188, 170, 127)
BUTTON_TEXT_COLOR = (47, 43, 35)
EMPTY_CELL_COLOR = (253, 251, 246)


@dataclass(frozen=True)
class WebButton:
    """Clickable browser control."""

    label: str
    action: str


@dataclass
class WebViewerConfig:
    cell_size: int = 26
    margin: int = 20
    sidebar_width: int = 280
    footer_height: int = 180
    autoplay_delay_ms: int = 325
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        board_pixels = BOARD_SIZE * self.cell_size
        if self.width <= 0:
            self.width = board_pixels + self.sidebar_width + self.margin * 3
        if self.height <= 0:
            self.height = board_pixels + self.footer_height + self.margin * 3


class WebBlokusViewer:
    """Simple async viewer that works well with pygbag in the browser."""

    def __init__(self, config: WebViewerConfig | None = None) -> None:
        self.config = WebViewerConfig() if config is None else config
        self.buttons = [
            WebButton("Step", "step"),
            WebButton("Play", "toggle_autoplay"),
            WebButton("Reset", "reset"),
            WebButton("New Game", "new_game"),
        ]
        self.agents = self._build_agents()
        self.state = GameState.new_game(player_count=len(self.agents))
        self.turn_count = 0
        self.pass_count = 0
        self.consecutive_passes = 0
        self.autoplay = True
        self.status_message = "Autoplaying a four-agent match."
        self.selected_seed = 0
        self.last_move: Move | None = None
        self.winners: list[int] = []
        self._finished = False
        self._autoplay_accumulator_ms = 0
        self._crash_lines: list[str] = []

    async def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "pygame is required for the web viewer. Install it with `pip install pygame pygbag`."
            ) from exc

        pygame.init()
        pygame.font.init()

        screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Blokus AI Web Viewer")
        clock = pygame.time.Clock()
        title_font = pygame.font.Font(None, 34)
        body_font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 20)

        running = True
        while running:
            try:
                elapsed_ms = clock.tick(60)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        running = self._handle_key(event.key, pygame)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self._handle_click(event.pos, pygame)

                if self.autoplay and not self._finished:
                    self._autoplay_accumulator_ms += elapsed_ms
                    if self._autoplay_accumulator_ms >= self.config.autoplay_delay_ms:
                        self._advance_one_turn()
                        self._autoplay_accumulator_ms = 0

                self._draw(screen, title_font, body_font, small_font, pygame)
            except Exception:
                self._crash_lines = traceback.format_exc().splitlines()
                self.autoplay = False
                self._finished = True
                self._draw_crash(screen, title_font, small_font, pygame)
            pygame.display.flip()
            await asyncio.sleep(0)

        pygame.quit()

    def _handle_key(self, key, pygame) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_SPACE:
            self._advance_one_turn()
        elif key == pygame.K_p:
            self._toggle_autoplay()
        elif key == pygame.K_r:
            self._reset_match()
        elif key == pygame.K_n:
            self._new_game()
        return True

    def _handle_click(self, position: tuple[int, int], pygame) -> None:
        for action, rect in self._button_rects(pygame).items():
            if rect.collidepoint(position):
                self._run_action(action)
                break

    def _run_action(self, action: str) -> None:
        if action == "step":
            self._advance_one_turn()
        elif action == "toggle_autoplay":
            self._toggle_autoplay()
        elif action == "reset":
            self._reset_match()
        elif action == "new_game":
            self._new_game()

    def _toggle_autoplay(self) -> None:
        self.autoplay = not self.autoplay
        self.status_message = "Autoplay enabled." if self.autoplay else "Autoplay paused."

    def _reset_match(self) -> None:
        self.state = GameState.new_game(player_count=len(self.agents))
        self.turn_count = 0
        self.pass_count = 0
        self.consecutive_passes = 0
        self.autoplay = False
        self.status_message = "Match reset. Press Play or Step."
        self.last_move = None
        self.winners = []
        self._finished = False
        self._autoplay_accumulator_ms = 0

    def _new_game(self) -> None:
        self.selected_seed += 1
        self.agents = self._build_agents()
        for index, agent in enumerate(self.agents):
            if isinstance(agent, RandomAgent):
                agent.rng.seed(1000 + self.selected_seed + index)
        self._reset_match()
        self.autoplay = True
        self.status_message = f"Started game #{self.selected_seed + 1}."

    def _advance_one_turn(self) -> None:
        if self._finished:
            return

        player = self.state.current_player
        legal_moves = generate_legal_moves(self.state, player=player)
        if legal_moves:
            move = self.agents[player].select_move(self.state, legal_moves)
            if move is None or move not in legal_moves:
                raise ValueError(f"Agent for player {player} returned an invalid move.")
            self.state = self.state.apply_move(move)
            self.turn_count += 1
            self.consecutive_passes = 0
            self.last_move = move
            self.status_message = f"{self._player_label(player)} played {move.piece_name}."
        else:
            self.state = self.state.pass_turn()
            self.turn_count += 1
            self.pass_count += 1
            self.consecutive_passes += 1
            self.last_move = None
            self.status_message = f"{self._player_label(player)} passed."

        if self.consecutive_passes >= len(self.state.remaining_pieces):
            self._finished = True
            self.autoplay = False
            scores = self.state.scores()
            winning_score = max(scores.values()) if scores else 0
            self.winners = [entry for entry, score in scores.items() if score == winning_score]
            winner_names = ", ".join(self._player_label(player) for player in self.winners)
            self.status_message = f"Game over. Winner: {winner_names}."

    def _draw(self, screen, title_font, body_font, small_font, pygame) -> None:
        screen.fill(BACKGROUND_COLOR)
        board_left = self.config.margin
        board_top = self.config.margin
        board_size = BOARD_SIZE * self.config.cell_size

        self._draw_board(screen, board_left, board_top, pygame)
        self._draw_sidebar(
            screen,
            board_left + board_size + self.config.margin,
            board_top,
            board_size,
            title_font,
            body_font,
            small_font,
            pygame,
        )
        self._draw_footer(
            screen,
            board_left,
            board_top + board_size + self.config.margin,
            board_size,
            body_font,
            small_font,
            pygame,
        )

    def _draw_crash(self, screen, title_font, small_font, pygame) -> None:
        screen.fill((36, 31, 28))
        panel = pygame.Rect(24, 24, self.config.width - 48, self.config.height - 48)
        pygame.draw.rect(screen, (245, 233, 228), panel, border_radius=16)
        pygame.draw.rect(screen, (150, 82, 70), panel, width=3, border_radius=16)

        heading = title_font.render("Blokus web viewer crashed", True, (120, 40, 28))
        screen.blit(heading, (40, 40))

        y = 88
        for line in self._crash_lines[:20]:
            surface = small_font.render(line[:140], True, (50, 45, 42))
            screen.blit(surface, (40, y))
            y += 22

    def _draw_board(self, screen, board_left: int, board_top: int, pygame) -> None:
        board_rect = pygame.Rect(
            board_left - 4,
            board_top - 4,
            BOARD_SIZE * self.config.cell_size + 8,
            BOARD_SIZE * self.config.cell_size + 8,
        )
        pygame.draw.rect(screen, PANEL_BORDER, board_rect, border_radius=12)
        pygame.draw.rect(screen, BOARD_BACKGROUND, board_rect.inflate(-6, -6), border_radius=10)

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
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRID_COLOR, rect, width=1)

    def _draw_sidebar(self, screen, left: int, top: int, board_pixels: int, title_font, body_font, small_font, pygame) -> None:
        panel_rect = pygame.Rect(left, top, self.config.sidebar_width, board_pixels)
        pygame.draw.rect(screen, PANEL_COLOR, panel_rect, border_radius=14)
        pygame.draw.rect(screen, PANEL_BORDER, panel_rect, width=2, border_radius=14)

        y = top + 18
        title_surface = title_font.render("Blokus AI", True, TEXT_COLOR)
        screen.blit(title_surface, (left + 16, y))
        y += 40

        lines = [
            f"Turn: {self.turn_count}",
            f"Current: {self._player_label(self.state.current_player)}",
            f"Passes: {self.pass_count}",
            f"Status: {'Finished' if self._finished else ('Autoplay' if self.autoplay else 'Paused')}",
            f"Last move: {self._last_move_label()}",
            "",
            "Players",
        ]
        lines.extend(
            f"{self._player_label(player)}: {self.state.scores().get(player, 0)}"
            for player in sorted(self.state.remaining_pieces)
        )

        for line in lines:
            font = body_font if line in {"Players"} else small_font
            if line == "":
                y += 8
                continue
            surface = font.render(line, True, TEXT_COLOR)
            screen.blit(surface, (left + 16, y))
            y += 28 if font is body_font else 24

        y += 10
        for action, rect in self._button_rects(pygame).items():
            label = next(button.label for button in self.buttons if button.action == action)
            active = action == "toggle_autoplay" and self.autoplay
            fill = BUTTON_ACTIVE_COLOR if active else BUTTON_COLOR
            pygame.draw.rect(screen, fill, rect, border_radius=10)
            pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)
            surface = small_font.render(label, True, BUTTON_TEXT_COLOR)
            text_rect = surface.get_rect(center=rect.center)
            screen.blit(surface, text_rect)

    def _draw_footer(self, screen, left: int, top: int, board_pixels: int, body_font, small_font, pygame) -> None:
        footer_width = board_pixels + self.config.sidebar_width + self.config.margin
        footer_rect = pygame.Rect(left, top, footer_width, self.config.footer_height)
        pygame.draw.rect(screen, PANEL_COLOR, footer_rect, border_radius=14)
        pygame.draw.rect(screen, PANEL_BORDER, footer_rect, width=2, border_radius=14)

        heading = body_font.render("Browser Controls", True, TEXT_COLOR)
        screen.blit(heading, (left + 16, top + 14))

        controls = [
            "Mouse: click Step, Play, Reset, or New Game",
            "Keyboard: SPACE step, P play/pause, R reset, N new game, ESC quit",
            "Matchup: Random vs Largest vs Weighted vs Adaptive",
            self.status_message,
        ]
        y = top + 52
        for line in controls:
            surface = small_font.render(line, True, TEXT_COLOR)
            screen.blit(surface, (left + 16, y))
            y += 26

    def _button_rects(self, pygame) -> dict[str, object]:
        board_pixels = BOARD_SIZE * self.config.cell_size
        panel_left = self.config.margin + board_pixels + self.config.margin
        top = self.config.margin + 280
        width = self.config.sidebar_width - 32
        height = 38
        spacing = 12
        return {
            button.action: pygame.Rect(panel_left + 16, top + index * (height + spacing), width, height)
            for index, button in enumerate(self.buttons)
        }

    def _build_agents(self) -> list[Agent]:
        return [
            RandomAgent(),
            LargestFirstAgent(),
            WeightedBlockingAgent(blocked_corner_weight=0.4),
            AdaptiveWeightedBlockingAgent(
                early_block_weight=0.15,
                late_block_weight=1.0,
                rank_weights=(1.0, 0.6, 0.25),
            ),
        ]

    def _player_label(self, player: int) -> str:
        labels = {
            0: "P0 Random",
            1: "P1 Largest",
            2: "P2 Weighted",
            3: "P3 Adaptive",
        }
        return labels.get(player, f"Player {player}")

    def _last_move_label(self) -> str:
        if self.last_move is None:
            return "pass" if self.turn_count > 0 else "none"
        return f"{self.last_move.piece_name} @ {self.last_move.origin}"


async def run_web_viewer() -> None:
    """Launch the browser-oriented viewer."""
    await WebBlokusViewer().run()

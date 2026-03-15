"""Minimal pygame viewer for Blokus self-play and move replay."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.core.board import BOARD_SIZE
from blokus_ai.core.game_state import GameState
from blokus_ai.core.move import Move
from blokus_ai.experiments.self_play import SelfPlaySession, play_random_game

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


@dataclass
class ViewerConfig:
    cell_size: int = 28
    margin: int = 24
    sidebar_width: int = 220
    autoplay_delay_ms: int = 250
    window_title: str = "Blokus Viewer"


class PygameViewer:
    """Simple viewer that can step through live self-play or a replay move list."""

    def __init__(
        self,
        session_factory: Callable[[], SelfPlaySession],
        config: ViewerConfig | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.config = ViewerConfig() if config is None else config
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
        height = board_pixels + self.config.margin * 2

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
            "Scores:",
        ]
        lines.extend(f"Player {player}: {score}" for player, score in sorted(scores.items()))
        lines.extend(
            [
                "Controls:",
                "SPACE step",
                "P autoplay",
                "R reset",
                "ESC quit",
            ]
        )
        return lines


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
    moves: list[Move],
    initial_state: GameState | None = None,
    player_count: int = 4,
) -> None:
    """Launch the pygame viewer in replay mode for a precomputed move list."""

    def session_factory() -> SelfPlaySession:
        return SelfPlaySession.from_move_list(
            replay_moves=moves,
            initial_state=initial_state,
            player_count=player_count,
        )

    PygameViewer(session_factory=session_factory).run()


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
    run_move_replay_viewer(result.moves)


if __name__ == "__main__":
    main()

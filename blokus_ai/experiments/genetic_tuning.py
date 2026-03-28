"""Genetic search for strategic heuristic agent parameters."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from blokus_ai.agents.adaptive_weighted_blocking_agent import AdaptiveWeightedBlockingAgent
from blokus_ai.agents.base import Agent
from blokus_ai.agents.blocking_agent import BlockingAgent
from blokus_ai.agents.largest_first_agent import LargestFirstAgent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.agents.strategic_heuristic_agent import StrategicHeuristicAgent
from blokus_ai.agents.weighted_blocking_agent import WeightedBlockingAgent

from .self_play import play_game

GENERATIONS = 30
POPULATION_SIZE = 18
ELITE_COUNT = 3
RANDOM_IMMIGRANTS = 2
TOURNAMENT_SIZE = 3
GAMES_PER_GENOME = 16
INITIAL_SEED = 123
MUTATION_SCALE = 0.18
GENOMES_PER_MATCH = 2


@dataclass(frozen=True)
class StrategicGenome:
    """Parameter bundle for one strategic heuristic agent."""

    name: str
    piece_weight: float
    early_block_weight: float
    late_block_weight: float
    own_frontier_weight: float
    mobility_weight: float
    early_piece_pressure_weight: float
    late_piece_pressure_weight: float
    rank_weights: tuple[float, float, float]

    def build_agent(self) -> StrategicHeuristicAgent:
        return StrategicHeuristicAgent(
            piece_weight=self.piece_weight,
            early_block_weight=self.early_block_weight,
            late_block_weight=self.late_block_weight,
            own_frontier_weight=self.own_frontier_weight,
            mobility_weight=self.mobility_weight,
            early_piece_pressure_weight=self.early_piece_pressure_weight,
            late_piece_pressure_weight=self.late_piece_pressure_weight,
            rank_weights=self.rank_weights,
        )


AdaptiveGenome = StrategicGenome


@dataclass(frozen=True)
class GenomeEvaluation:
    """Aggregate results for one genome in a generation."""

    genome: StrategicGenome
    games_played: int
    average_score: float
    average_score_margin: float
    win_points: float
    win_rate: float


@dataclass(frozen=True)
class GenerationResult:
    """Ranked results for one generation."""

    generation_index: int
    rankings: list[GenomeEvaluation]
    average_game_length: float
    matches_played: int


@dataclass(frozen=True)
class BaselineEntry:
    """Fixed non-evolving opponent used to improve robustness during tuning."""

    name: str
    prototype: Agent
    weight: float = 1.0


def random_genome(name: str, rng: random.Random) -> StrategicGenome:
    return _normalize_genome(
        name=name,
        piece_weight=rng.uniform(0.7, 1.4),
        early_block_weight=rng.uniform(0.0, 0.5),
        late_block_weight=rng.uniform(0.5, 2.0),
        own_frontier_weight=rng.uniform(0.0, 0.5),
        mobility_weight=rng.uniform(0.0, 0.12),
        early_piece_pressure_weight=rng.uniform(0.0, 0.9),
        late_piece_pressure_weight=rng.uniform(0.0, 0.5),
        rank_weights=tuple(rng.uniform(0.1, 2.2) for _ in range(3)),
    )


def seed_population(population_size: int, rng: random.Random) -> list[StrategicGenome]:
    seeded = [
        _normalize_genome(
            name="g0_seed_strategic_default",
            piece_weight=1.0,
            early_block_weight=0.15,
            late_block_weight=1.05,
            own_frontier_weight=0.18,
            mobility_weight=0.04,
            early_piece_pressure_weight=0.55,
            late_piece_pressure_weight=0.1,
            rank_weights=(1.5, 0.8, 0.3),
        ),
        _normalize_genome(
            name="g0_seed_block_heavy",
            piece_weight=1.0,
            early_block_weight=0.15,
            late_block_weight=1.8,
            own_frontier_weight=0.1,
            mobility_weight=0.02,
            early_piece_pressure_weight=0.45,
            late_piece_pressure_weight=0.05,
            rank_weights=(1.6, 0.9, 0.4),
        ),
        _normalize_genome(
            name="g0_seed_expansion_heavy",
            piece_weight=1.05,
            early_block_weight=0.05,
            late_block_weight=0.8,
            own_frontier_weight=0.35,
            mobility_weight=0.08,
            early_piece_pressure_weight=0.7,
            late_piece_pressure_weight=0.18,
            rank_weights=(1.2, 0.7, 0.3),
        ),
    ]

    while len(seeded) < population_size:
        seeded.append(random_genome(name=f"g0_rand_{len(seeded)}", rng=rng))
    return seeded[:population_size]


def crossover_genomes(name: str, first: StrategicGenome, second: StrategicGenome) -> StrategicGenome:
    """Blend parents with per-gene interpolation to preserve diversity."""
    rng = random.Random(f"{name}:{first.name}:{second.name}")

    def blend(value_a: float, value_b: float) -> float:
        alpha = rng.random()
        return value_a * alpha + value_b * (1.0 - alpha)

    return _normalize_genome(
        name=name,
        piece_weight=blend(first.piece_weight, second.piece_weight),
        early_block_weight=blend(first.early_block_weight, second.early_block_weight),
        late_block_weight=blend(first.late_block_weight, second.late_block_weight),
        own_frontier_weight=blend(first.own_frontier_weight, second.own_frontier_weight),
        mobility_weight=blend(first.mobility_weight, second.mobility_weight),
        early_piece_pressure_weight=blend(
            first.early_piece_pressure_weight,
            second.early_piece_pressure_weight,
        ),
        late_piece_pressure_weight=blend(
            first.late_piece_pressure_weight,
            second.late_piece_pressure_weight,
        ),
        rank_weights=tuple(
            blend(first.rank_weights[index], second.rank_weights[index])
            for index in range(3)
        ),
    )


def mutate_genome(
    name: str,
    parent: StrategicGenome,
    rng: random.Random,
    scale: float = MUTATION_SCALE,
) -> StrategicGenome:
    """Apply gaussian perturbations to one parent's genes."""
    return _normalize_genome(
        name=name,
        piece_weight=parent.piece_weight + rng.gauss(0.0, scale * 0.6),
        early_block_weight=parent.early_block_weight + rng.gauss(0.0, scale),
        late_block_weight=parent.late_block_weight + rng.gauss(0.0, scale),
        own_frontier_weight=parent.own_frontier_weight + rng.gauss(0.0, scale * 0.7),
        mobility_weight=parent.mobility_weight + rng.gauss(0.0, scale * 0.18),
        early_piece_pressure_weight=parent.early_piece_pressure_weight + rng.gauss(0.0, scale),
        late_piece_pressure_weight=parent.late_piece_pressure_weight + rng.gauss(0.0, scale * 0.7),
        rank_weights=tuple(weight + rng.gauss(0.0, scale) for weight in parent.rank_weights),
    )


def build_baseline_pool() -> list[BaselineEntry]:
    """Fixed opponents that genomes must handle well during evaluation."""
    return [
        BaselineEntry("BlockingAgent", BlockingAgent(), weight=4.0),
        BaselineEntry("BlockingAgent", BlockingAgent(), weight=4.0),
        BaselineEntry("WeightedBlockingAgent(1/3)", WeightedBlockingAgent(blocked_corner_weight=1.0 / 3.0), weight=2.0),
        BaselineEntry("LargestFirstAgent", LargestFirstAgent(), weight=1.5),
        BaselineEntry(
            "AdaptiveWeightedBlockingAgent",
            AdaptiveWeightedBlockingAgent(
                early_block_weight=0.1,
                late_block_weight=1.0,
                rank_weights=(1.0, 0.6, 0.3),
            ),
            weight=2.0,
        ),
        BaselineEntry("StrategicHeuristicAgent", StrategicHeuristicAgent(), weight=2.5),
        BaselineEntry("RandomAgent", RandomAgent(), weight=0.3),
    ]


def evaluate_population(
    population: list[StrategicGenome],
    games_per_genome: int,
    seed: int,
    baseline_pool: list[BaselineEntry] | None = None,
    genomes_per_match: int = GENOMES_PER_MATCH,
    verbose: bool = False,
    generation_index: int | None = None,
) -> GenerationResult:
    """Play sampled 4-player matches and rank genomes by wins, margin, then score."""
    if len(population) < genomes_per_match:
        raise ValueError("Population size must be at least genomes_per_match.")
    if games_per_genome <= 0:
        raise ValueError("games_per_genome must be positive.")
    if genomes_per_match <= 0 or genomes_per_match > 4:
        raise ValueError("genomes_per_match must be between 1 and 4.")

    baseline_pool = build_baseline_pool() if baseline_pool is None else baseline_pool
    baseline_slots = 4 - genomes_per_match
    if baseline_slots > 0 and not baseline_pool:
        raise ValueError("A non-empty baseline_pool is required when genomes_per_match < 4.")

    rng = random.Random(seed)
    scheduled_groups = _scheduled_genome_groups(
        population=population,
        games_per_genome=games_per_genome,
        genomes_per_match=genomes_per_match,
        rng=rng,
    )
    matches_to_play = len(scheduled_groups)
    total_scores = {genome.name: 0.0 for genome in population}
    total_score_margins = {genome.name: 0.0 for genome in population}
    win_points = {genome.name: 0.0 for genome in population}
    games_played = {genome.name: 0 for genome in population}
    total_turns = 0

    for match_index, selected_genomes in enumerate(scheduled_groups, start=1):
        if verbose and (
            match_index == 1
            or match_index == matches_to_play
            or match_index % max(1, matches_to_play // 5) == 0
        ):
            generation_label = generation_index if generation_index is not None else "?"
            print(f"[ga] generation={generation_label} match={match_index}/{matches_to_play}")

        selected_baselines = _sample_baselines(baseline_pool, baseline_slots, rng)
        seating_entries: list[StrategicGenome | BaselineEntry] = selected_genomes + selected_baselines
        rng.shuffle(seating_entries)
        agents = [_build_agent(entry) for entry in seating_entries]
        result = play_game(agents=agents, print_boards=False)
        total_turns += result.turn_count

        seat_to_genome = {
            seat: entry for seat, entry in enumerate(seating_entries) if isinstance(entry, StrategicGenome)
        }
        for seat, score in result.scores.items():
            genome = seat_to_genome.get(seat)
            if genome is None:
                continue
            opponent_scores = [
                other_score
                for other_seat, other_score in result.scores.items()
                if other_seat != seat
            ]
            total_scores[genome.name] += score
            total_score_margins[genome.name] += score - (sum(opponent_scores) / len(opponent_scores))
            games_played[genome.name] += 1

        shared_win_value = 1.0 / len(result.winners)
        for seat in result.winners:
            genome = seat_to_genome.get(seat)
            if genome is not None:
                win_points[genome.name] += shared_win_value

    evaluations = [
        GenomeEvaluation(
            genome=genome,
            games_played=games_played[genome.name],
            average_score=(
                total_scores[genome.name] / games_played[genome.name]
                if games_played[genome.name] > 0
                else 0.0
            ),
            average_score_margin=(
                total_score_margins[genome.name] / games_played[genome.name]
                if games_played[genome.name] > 0
                else 0.0
            ),
            win_points=win_points[genome.name],
            win_rate=(
                win_points[genome.name] / games_played[genome.name]
                if games_played[genome.name] > 0
                else 0.0
            ),
        )
        for genome in population
    ]
    rankings = sorted(
        evaluations,
        key=lambda entry: (
            -entry.win_rate,
            -entry.average_score_margin,
            -entry.average_score,
            entry.genome.name,
        ),
    )

    return GenerationResult(
        generation_index=0,
        rankings=rankings,
        average_game_length=total_turns / matches_to_play,
        matches_played=matches_to_play,
    )


def evolve_population(
    generations: int = GENERATIONS,
    population_size: int = POPULATION_SIZE,
    elite_count: int = ELITE_COUNT,
    games_per_genome: int = GAMES_PER_GENOME,
    seed: int = INITIAL_SEED,
    baseline_pool: list[BaselineEntry] | None = None,
    genomes_per_match: int = GENOMES_PER_MATCH,
    random_immigrants: int = RANDOM_IMMIGRANTS,
    tournament_size: int = TOURNAMENT_SIZE,
    verbose: bool = True,
) -> list[GenerationResult]:
    """Run the GA loop with elitism, tournament selection, and random immigrants."""
    if population_size < genomes_per_match:
        raise ValueError("population_size must be at least genomes_per_match.")
    if elite_count <= 0 or elite_count >= population_size:
        raise ValueError("elite_count must be positive and less than population_size.")
    if random_immigrants < 0 or elite_count + random_immigrants >= population_size:
        raise ValueError("elite_count plus random_immigrants must be less than population_size.")

    rng = random.Random(seed)
    population = seed_population(population_size=population_size, rng=rng)
    history: list[GenerationResult] = []

    if verbose:
        print(
            f"[ga] start generations={generations} population_size={population_size} "
            f"elite_count={elite_count} random_immigrants={random_immigrants} "
            f"games_per_genome={games_per_genome} genomes_per_match={genomes_per_match}"
        )

    for generation_index in range(generations):
        generation_seed = rng.randrange(0, 2**32)
        if verbose:
            print(f"[ga] generation={generation_index} evaluating population")
        result = evaluate_population(
            population,
            games_per_genome=games_per_genome,
            seed=generation_seed,
            baseline_pool=baseline_pool,
            genomes_per_match=genomes_per_match,
            verbose=verbose,
            generation_index=generation_index,
        )
        result = GenerationResult(
            generation_index=generation_index,
            rankings=result.rankings,
            average_game_length=result.average_game_length,
            matches_played=result.matches_played,
        )
        history.append(result)

        if verbose:
            leader = result.rankings[0]
            print(
                f"[ga] generation={generation_index} done "
                f"leader={leader.genome.name} win_rate={leader.win_rate:.3f} "
                f"avg_margin={leader.average_score_margin:.2f} avg_score={leader.average_score:.2f}"
            )

        evaluations = result.rankings
        elites = [entry.genome for entry in evaluations[:elite_count]]
        next_population = list(elites)

        for immigrant_index in range(random_immigrants):
            next_population.append(
                random_genome(
                    name=f"g{generation_index + 1}_immigrant_{immigrant_index}",
                    rng=rng,
                )
            )

        child_index = 0
        while len(next_population) < population_size:
            first_parent = _tournament_select(evaluations, rng, tournament_size).genome
            second_parent = _tournament_select(evaluations, rng, tournament_size).genome

            if rng.random() < 0.55:
                child = crossover_genomes(
                    name=f"g{generation_index + 1}_cross_{child_index}",
                    first=first_parent,
                    second=second_parent,
                )
                if rng.random() < 0.7:
                    child = mutate_genome(
                        name=f"g{generation_index + 1}_crossmut_{child_index}",
                        parent=child,
                        rng=rng,
                    )
            else:
                child = mutate_genome(
                    name=f"g{generation_index + 1}_mut_{child_index}",
                    parent=first_parent,
                    rng=rng,
                )

            next_population.append(child)
            child_index += 1

        population = next_population

        if verbose:
            print(f"[ga] generation={generation_index} bred next population\n")

    return history


def main() -> None:
    history = evolve_population(verbose=True)

    for generation in history:
        print(
            f"Generation {generation.generation_index}: "
            f"matches={generation.matches_played} | "
            f"avg_game_length={generation.average_game_length:.2f} turns"
        )
        for place, entry in enumerate(generation.rankings[:8], start=1):
            genome = entry.genome
            print(
                f"  {place}. {genome.name} | "
                f"games={entry.games_played} | "
                f"win_rate={entry.win_rate:.3f} | "
                f"avg_margin={entry.average_score_margin:.2f} | "
                f"avg_score={entry.average_score:.2f} | "
                f"piece={genome.piece_weight:.3f} | "
                f"block=({genome.early_block_weight:.3f},{genome.late_block_weight:.3f}) | "
                f"frontier={genome.own_frontier_weight:.3f} | "
                f"mobility={genome.mobility_weight:.3f} | "
                f"piece_pressure=({genome.early_piece_pressure_weight:.3f},{genome.late_piece_pressure_weight:.3f}) | "
                f"rank_weights={tuple(round(weight, 3) for weight in genome.rank_weights)}"
            )
        print()

    champion = history[-1].rankings[0].genome
    print("Champion genome:")
    print(
        f"  {champion.name} | piece={champion.piece_weight:.3f} | "
        f"block=({champion.early_block_weight:.3f},{champion.late_block_weight:.3f}) | "
        f"frontier={champion.own_frontier_weight:.3f} | "
        f"mobility={champion.mobility_weight:.3f} | "
        f"piece_pressure=({champion.early_piece_pressure_weight:.3f},{champion.late_piece_pressure_weight:.3f}) | "
        f"rank_weights={tuple(round(weight, 3) for weight in champion.rank_weights)}"
    )


def _build_agent(entry: StrategicGenome | BaselineEntry) -> Agent:
    if isinstance(entry, StrategicGenome):
        return entry.build_agent()

    return copy.deepcopy(entry.prototype)


def _sample_baselines(
    baseline_pool: list[BaselineEntry],
    count: int,
    rng: random.Random,
) -> list[BaselineEntry]:
    if count <= 0:
        return []

    remaining = list(baseline_pool)
    selected: list[BaselineEntry] = []
    while remaining and len(selected) < count:
        total_weight = sum(entry.weight for entry in remaining)
        threshold = rng.uniform(0.0, total_weight)
        cumulative = 0.0
        for index, entry in enumerate(remaining):
            cumulative += entry.weight
            if cumulative >= threshold:
                selected.append(entry)
                remaining.pop(index)
                break

    while len(selected) < count:
        selected.append(rng.choice(baseline_pool))

    return selected


def _scheduled_genome_groups(
    population: list[StrategicGenome],
    games_per_genome: int,
    genomes_per_match: int,
    rng: random.Random,
) -> list[list[StrategicGenome]]:
    appearance_pool = [genome for genome in population for _ in range(games_per_genome)]
    rng.shuffle(appearance_pool)

    groups: list[list[StrategicGenome]] = []
    current: list[StrategicGenome] = []
    current_names: set[str] = set()

    for genome in appearance_pool:
        if genome.name in current_names or len(current) == genomes_per_match:
            groups.append(current)
            current = []
            current_names = set()

        current.append(genome)
        current_names.add(genome.name)

    if current:
        groups.append(current)

    for group in groups:
        while len(group) < genomes_per_match:
            candidate = rng.choice(population)
            if candidate.name in {genome.name for genome in group} and len(population) >= genomes_per_match:
                continue
            group.append(candidate)

    return groups


def _tournament_select(
    evaluations: list[GenomeEvaluation],
    rng: random.Random,
    tournament_size: int,
) -> GenomeEvaluation:
    sampled = rng.sample(evaluations, k=min(tournament_size, len(evaluations)))
    return min(
        sampled,
        key=lambda entry: (
            -entry.win_rate,
            -entry.average_score_margin,
            -entry.average_score,
            entry.genome.name,
        ),
    )


def _normalize_genome(
    name: str,
    piece_weight: float,
    early_block_weight: float,
    late_block_weight: float,
    own_frontier_weight: float,
    mobility_weight: float,
    early_piece_pressure_weight: float,
    late_piece_pressure_weight: float,
    rank_weights: tuple[float, float, float] | tuple[float, ...],
) -> StrategicGenome:
    clipped_rank_weights = tuple(
        sorted((max(0.0, weight) for weight in rank_weights[:3]), reverse=True)
    )
    while len(clipped_rank_weights) < 3:
        clipped_rank_weights = clipped_rank_weights + (
            clipped_rank_weights[-1] if clipped_rank_weights else 0.0,
        )

    clipped_piece_weight = max(0.1, piece_weight)
    clipped_early_block = max(0.0, early_block_weight)
    clipped_late_block = max(clipped_early_block, late_block_weight)

    return StrategicGenome(
        name=name,
        piece_weight=clipped_piece_weight,
        early_block_weight=clipped_early_block,
        late_block_weight=clipped_late_block,
        own_frontier_weight=max(0.0, own_frontier_weight),
        mobility_weight=max(0.0, mobility_weight),
        early_piece_pressure_weight=max(0.0, early_piece_pressure_weight),
        late_piece_pressure_weight=max(0.0, late_piece_pressure_weight),
        rank_weights=clipped_rank_weights,
    )

"""Genetic search for adaptive weighted blocking parameters."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from blokus_ai.agents.adaptive_weighted_blocking_agent import AdaptiveWeightedBlockingAgent
from blokus_ai.agents.base import Agent
from blokus_ai.agents.blocking_agent import BlockingAgent
from blokus_ai.agents.largest_first_agent import LargestFirstAgent
from blokus_ai.agents.random_agent import RandomAgent
from blokus_ai.agents.weighted_blocking_agent import WeightedBlockingAgent

from .self_play import play_game

GENERATIONS = 20
POPULATION_SIZE = 12
ELITE_COUNT = 3
GAMES_PER_GENOME = 10
INITIAL_SEED = 123
MUTATION_SCALE = 0.2
GENOMES_PER_MATCH = 2


@dataclass(frozen=True)
class AdaptiveGenome:
    """Parameter bundle for one adaptive weighted blocking agent."""

    name: str
    early_block_weight: float
    late_block_weight: float
    rank_weights: tuple[float, float, float]

    def build_agent(self) -> AdaptiveWeightedBlockingAgent:
        return AdaptiveWeightedBlockingAgent(
            early_block_weight=self.early_block_weight,
            late_block_weight=self.late_block_weight,
            rank_weights=self.rank_weights,
        )


@dataclass(frozen=True)
class GenomeEvaluation:
    """Aggregate results for one genome in a generation."""

    genome: AdaptiveGenome
    games_played: int
    average_score: float
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


def random_genome(name: str, rng: random.Random) -> AdaptiveGenome:
    early = rng.uniform(0.0, 0.6)
    late = rng.uniform(early, 2.0)
    rank_weights = tuple(sorted((rng.uniform(0.1, 2.0) for _ in range(3)), reverse=True))
    return AdaptiveGenome(
        name=name,
        early_block_weight=early,
        late_block_weight=late,
        rank_weights=rank_weights,
    )


def crossover_genomes(name: str, first: AdaptiveGenome, second: AdaptiveGenome) -> AdaptiveGenome:
    """Blend two parents by averaging their genes."""
    return _normalize_genome(
        name=name,
        early_block_weight=(first.early_block_weight + second.early_block_weight) / 2.0,
        late_block_weight=(first.late_block_weight + second.late_block_weight) / 2.0,
        rank_weights=tuple(
            (first.rank_weights[index] + second.rank_weights[index]) / 2.0
            for index in range(3)
        ),
    )


def mutate_genome(
    name: str,
    parent: AdaptiveGenome,
    rng: random.Random,
    scale: float = MUTATION_SCALE,
) -> AdaptiveGenome:
    """Apply small gaussian perturbations to one parent's genes."""
    return _normalize_genome(
        name=name,
        early_block_weight=parent.early_block_weight + rng.gauss(0.0, scale),
        late_block_weight=parent.late_block_weight + rng.gauss(0.0, scale),
        rank_weights=tuple(weight + rng.gauss(0.0, scale) for weight in parent.rank_weights),
    )


def build_baseline_pool() -> list[BaselineEntry]:
    """Fixed opponents that genomes must handle well during evaluation."""
    return [
        BaselineEntry("BlockingAgent", BlockingAgent()),
        BaselineEntry("LargestFirstAgent", LargestFirstAgent()),
        BaselineEntry("WeightedBlockingAgent(1/3)", WeightedBlockingAgent(blocked_corner_weight=1.0 / 3.0)),
        BaselineEntry(
            "AdaptiveWeightedBlockingAgent",
            AdaptiveWeightedBlockingAgent(
                early_block_weight=0.1,
                late_block_weight=1.0,
                rank_weights=(1.0, 0.6, 0.3),
            ),
        ),
        BaselineEntry("RandomAgent", RandomAgent()),
    ]


def evaluate_population(
    population: list[AdaptiveGenome],
    games_per_genome: int,
    seed: int,
    baseline_pool: list[BaselineEntry] | None = None,
    genomes_per_match: int = GENOMES_PER_MATCH,
) -> GenerationResult:
    """Play many sampled 4-player matches and rank genomes by win rate then average score."""
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
    win_points = {genome.name: 0.0 for genome in population}
    games_played = {genome.name: 0 for genome in population}
    total_turns = 0

    for selected_genomes in scheduled_groups:
        selected_baselines = _sample_baselines(baseline_pool, baseline_slots, rng)
        seating_entries: list[AdaptiveGenome | BaselineEntry] = selected_genomes + selected_baselines
        rng.shuffle(seating_entries)
        agents = [_build_agent(entry) for entry in seating_entries]
        result = play_game(agents=agents, print_boards=False)
        total_turns += result.turn_count

        seat_to_genome = {
            seat: entry for seat, entry in enumerate(seating_entries) if isinstance(entry, AdaptiveGenome)
        }
        for seat, score in result.scores.items():
            genome = seat_to_genome.get(seat)
            if genome is None:
                continue
            total_scores[genome.name] += score
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
        key=lambda entry: (-entry.win_rate, -entry.average_score, entry.genome.name),
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
) -> list[GenerationResult]:
    """Run the GA loop with elitism plus crossover/mutation to refill the population."""
    if population_size < genomes_per_match:
        raise ValueError("population_size must be at least genomes_per_match.")
    if elite_count <= 0 or elite_count >= population_size:
        raise ValueError("elite_count must be positive and less than population_size.")

    rng = random.Random(seed)
    population = [random_genome(name=f"g0_p{index}", rng=rng) for index in range(population_size)]
    history: list[GenerationResult] = []

    for generation_index in range(generations):
        generation_seed = rng.randrange(0, 2**32)
        result = evaluate_population(
            population,
            games_per_genome=games_per_genome,
            seed=generation_seed,
            baseline_pool=baseline_pool,
            genomes_per_match=genomes_per_match,
        )
        result = GenerationResult(
            generation_index=generation_index,
            rankings=result.rankings,
            average_game_length=result.average_game_length,
            matches_played=result.matches_played,
        )
        history.append(result)

        elites = [entry.genome for entry in result.rankings[:elite_count]]
        next_population = list(elites)
        child_index = 0

        while len(next_population) < population_size:
            if len(next_population) == elite_count or rng.random() < 0.5:
                first, second = rng.sample(elites, k=2) if len(elites) >= 2 else (elites[0], elites[0])
                child = crossover_genomes(
                    name=f"g{generation_index + 1}_cross_{child_index}",
                    first=first,
                    second=second,
                )
            else:
                parent = rng.choice(elites)
                child = mutate_genome(
                    name=f"g{generation_index + 1}_mut_{child_index}",
                    parent=parent,
                    rng=rng,
                )
            next_population.append(child)
            child_index += 1

        population = next_population

    return history


def main() -> None:
    history = evolve_population()

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
                f"avg_score={entry.average_score:.2f} | "
                f"early={genome.early_block_weight:.3f} | "
                f"late={genome.late_block_weight:.3f} | "
                f"rank_weights={tuple(round(weight, 3) for weight in genome.rank_weights)}"
            )
        print()

    champion = history[-1].rankings[0].genome
    print("Champion genome:")
    print(
        f"  {champion.name} | early={champion.early_block_weight:.3f} | "
        f"late={champion.late_block_weight:.3f} | "
        f"rank_weights={tuple(round(weight, 3) for weight in champion.rank_weights)}"
    )


def _build_agent(entry: AdaptiveGenome | BaselineEntry) -> Agent:
    if isinstance(entry, AdaptiveGenome):
        return entry.build_agent()

    return copy.deepcopy(entry.prototype)


def _sample_baselines(
    baseline_pool: list[BaselineEntry],
    count: int,
    rng: random.Random,
) -> list[BaselineEntry]:
    if count <= 0:
        return []
    if len(baseline_pool) >= count:
        return list(rng.sample(baseline_pool, k=count))
    return [rng.choice(baseline_pool) for _ in range(count)]


def _scheduled_genome_groups(
    population: list[AdaptiveGenome],
    games_per_genome: int,
    genomes_per_match: int,
    rng: random.Random,
) -> list[list[AdaptiveGenome]]:
    appearance_pool = [genome for genome in population for _ in range(games_per_genome)]
    rng.shuffle(appearance_pool)

    groups: list[list[AdaptiveGenome]] = []
    current: list[AdaptiveGenome] = []
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


def _normalize_genome(
    name: str,
    early_block_weight: float,
    late_block_weight: float,
    rank_weights: tuple[float, float, float] | tuple[float, ...],
) -> AdaptiveGenome:
    clipped_early = max(0.0, early_block_weight)
    clipped_late = max(clipped_early, late_block_weight)
    clipped_rank_weights = tuple(
        sorted((max(0.0, weight) for weight in rank_weights[:3]), reverse=True)
    )
    while len(clipped_rank_weights) < 3:
        clipped_rank_weights = clipped_rank_weights + (
            clipped_rank_weights[-1] if clipped_rank_weights else 0.0,
        )

    return AdaptiveGenome(
        name=name,
        early_block_weight=clipped_early,
        late_block_weight=clipped_late,
        rank_weights=(
            clipped_rank_weights[0],
            clipped_rank_weights[1],
            clipped_rank_weights[2],
        ),
    )


if __name__ == "__main__":
    main()

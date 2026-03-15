"""Genetic search for adaptive weighted blocking parameters."""

from __future__ import annotations

import random
from dataclasses import dataclass

from blokus_ai.agents.adaptive_weighted_blocking_agent import AdaptiveWeightedBlockingAgent

from .self_play import play_game

GENERATIONS = 20
POPULATION_SIZE = 12
ELITE_COUNT = 3
GAMES_PER_GENOME = 10
INITIAL_SEED = 123
MUTATION_SCALE = 0.2


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


def evaluate_population(
    population: list[AdaptiveGenome],
    games_per_genome: int,
    seed: int,
) -> GenerationResult:
    """Play many sampled 4-player matches and rank genomes by win rate then average score."""
    if len(population) < 4:
        raise ValueError("Population size must be at least 4.")
    if games_per_genome <= 0:
        raise ValueError("games_per_genome must be positive.")

    rng = random.Random(seed)
    matches_to_play = max(1, (len(population) * games_per_genome + 3) // 4)
    total_scores = {genome.name: 0.0 for genome in population}
    win_points = {genome.name: 0.0 for genome in population}
    games_played = {genome.name: 0 for genome in population}
    total_turns = 0

    for _ in range(matches_to_play):
        seating = rng.sample(population, k=4)
        rng.shuffle(seating)
        agents = [genome.build_agent() for genome in seating]
        result = play_game(agents=agents, print_boards=False)
        total_turns += result.turn_count

        seat_to_genome = {seat: genome for seat, genome in enumerate(seating)}
        for seat, score in result.scores.items():
            genome_name = seat_to_genome[seat].name
            total_scores[genome_name] += score
            games_played[genome_name] += 1

        shared_win_value = 1.0 / len(result.winners)
        for seat in result.winners:
            win_points[seat_to_genome[seat].name] += shared_win_value

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
) -> list[GenerationResult]:
    """Run the GA loop with elitism plus crossover/mutation to refill the population."""
    if population_size < 4:
        raise ValueError("population_size must be at least 4.")
    if elite_count <= 0 or elite_count >= population_size:
        raise ValueError("elite_count must be positive and less than population_size.")

    rng = random.Random(seed)
    population = [random_genome(name=f"g0_p{index}", rng=rng) for index in range(population_size)]
    history: list[GenerationResult] = []

    for generation_index in range(generations):
        generation_seed = rng.randrange(0, 2**32)
        result = evaluate_population(population, games_per_genome=games_per_genome, seed=generation_seed)
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

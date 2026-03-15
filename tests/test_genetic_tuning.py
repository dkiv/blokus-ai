from blokus_ai.experiments.genetic_tuning import (
    AdaptiveGenome,
    BaselineEntry,
    build_baseline_pool,
    crossover_genomes,
    evaluate_population,
    mutate_genome,
)


def test_crossover_genome_preserves_monotone_weights() -> None:
    first = AdaptiveGenome("a", 0.1, 1.0, (1.0, 0.6, 0.3))
    second = AdaptiveGenome("b", 0.2, 0.8, (0.9, 0.2, 0.1))

    child = crossover_genomes("child", first, second)

    assert child.early_block_weight >= 0
    assert child.late_block_weight >= child.early_block_weight
    assert child.rank_weights[0] >= child.rank_weights[1] >= child.rank_weights[2]


def test_mutated_genome_preserves_nonnegative_monotone_weights() -> None:
    import random

    parent = AdaptiveGenome("parent", 0.1, 1.0, (1.0, 0.6, 0.3))
    child = mutate_genome("child", parent, rng=random.Random(1), scale=5.0)

    assert child.early_block_weight >= 0
    assert child.late_block_weight >= child.early_block_weight
    assert child.rank_weights[0] >= child.rank_weights[1] >= child.rank_weights[2]


def test_evaluate_population_tracks_per_genome_games_played() -> None:
    population = [
        AdaptiveGenome(f"g{i}", 0.1 + i * 0.01, 1.0 + i * 0.01, (1.0, 0.6, 0.3))
        for i in range(5)
    ]

    result = evaluate_population(population, games_per_genome=2, seed=1)

    assert result.matches_played >= 1
    assert len(result.rankings) == 5
    assert all(entry.games_played > 0 for entry in result.rankings)


def test_baseline_pool_is_available_for_mixed_matches() -> None:
    baseline_pool = build_baseline_pool()

    assert baseline_pool
    assert all(isinstance(entry, BaselineEntry) for entry in baseline_pool)

from blokus_ai.experiments.agent_comparison import build_agents, build_entries, build_labels


def test_agent_comparison_configuration_stays_aligned() -> None:
    assert len(build_agents()) == len(build_labels())
    assert len(build_entries()) == len(build_labels())

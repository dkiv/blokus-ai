"""Behavior cloning and actor-critic self-play training for Blokus."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim

from blokus_ai.agents import (
    AdaptiveWeightedBlockingAgent,
    BlockingAgent,
    LargestFirstAgent,
    RandomAgent,
    RLPolicyAgent,
    StrategicHeuristicAgent,
    WeightedBlockingAgent,
)
from blokus_ai.core.game_state import GameState
from blokus_ai.experiments.self_play import play_game
from blokus_ai.rl import BlokusRLEnvironment, encode_candidate_moves, encode_observation
from blokus_ai.rl.torch_policy import (
    TorchRLPolicy,
    batchify_policy_inputs,
    load_torch_policy,
    resolve_torch_device,
)

_MAX_SCORE = 89.0


@dataclass(frozen=True)
class RLSelfPlayStats:
    """Aggregate stats for an actor-critic training run."""

    episodes: int
    average_return: float
    average_turns: float
    policy_loss: float
    value_loss: float
    imitation_loss: float


@dataclass(frozen=True)
class RLEvalStats:
    """Aggregate results when evaluating a saved RL policy."""

    games: int
    average_score: float
    average_opponent_score: float
    win_rate: float
    average_turns: float


@dataclass(frozen=True)
class ImitationStats:
    """Aggregate results from behavior cloning warm start."""

    games: int
    examples: int
    epochs: int
    loss: float


@dataclass(frozen=True)
class ImitationExample:
    observation: Any
    candidates: tuple[Any, ...]
    action_index: int


@dataclass(frozen=True)
class RolloutStep:
    observation: Any
    candidates: tuple[Any, ...]
    action_index: int
    value_estimate: float
    reward: float


@dataclass(frozen=True)
class RolloutEpisode:
    steps: tuple[RolloutStep, ...]
    total_return: float
    turns: int


def train_actor_critic(
    episodes: int = 32,
    learning_rate: float = 1e-3,
    discount: float = 0.99,
    entropy_weight: float = 0.01,
    value_weight: float = 0.5,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    warm_start_games: int = 0,
    warm_start_epochs: int = 3,
    workers: int = 1,
    batch_episodes: int = 8,
) -> tuple[TorchRLPolicy, RLSelfPlayStats]:
    """Train an actor-critic policy from self-play, optionally with imitation warm start."""
    torch_device = resolve_torch_device(device)
    policy = TorchRLPolicy(device=torch_device)
    optimizer = optim.Adam(policy.network.parameters(), lr=learning_rate)

    imitation_loss = 0.0
    if warm_start_games > 0:
        _, imitation_stats = warm_start_policy(
            policy=policy,
            games=warm_start_games,
            epochs=warm_start_epochs,
            learning_rate=learning_rate,
            device=str(torch_device),
        )
        imitation_loss = imitation_stats.loss

    policy_losses: list[float] = []
    value_losses: list[float] = []
    episode_returns: list[float] = []
    episode_turns: list[int] = []

    episodes_remaining = episodes
    while episodes_remaining > 0:
        current_batch = min(batch_episodes, episodes_remaining)
        rollouts = _collect_rollouts(
            policy=policy,
            episodes=current_batch,
            discount=discount,
            workers=workers,
        )
        if not rollouts:
            break

        optimizer.zero_grad()
        total_policy_loss = torch.tensor(0.0, device=torch_device)
        total_value_loss = torch.tensor(0.0, device=torch_device)
        batch_weight = 0

        for episode in rollouts:
            returns = _discounted_returns(
                [step.reward for step in episode.steps],
                discount=discount,
                device=torch_device,
                normalize=False,
            )
            for step_index, rollout_step in enumerate(episode.steps):
                batch = batchify_policy_inputs(
                    rollout_step.observation,
                    rollout_step.candidates,
                    device=torch_device,
                )
                logits, values = policy.network(batch)
                distribution = torch.distributions.Categorical(logits=logits)
                action = torch.tensor(rollout_step.action_index, device=torch_device)
                log_prob = distribution.log_prob(action)
                entropy = distribution.entropy()
                value = values.squeeze(0)
                target_return = returns[step_index]
                advantage = (target_return - value).detach()

                total_policy_loss = total_policy_loss - (log_prob * advantage + entropy_weight * entropy)
                total_value_loss = total_value_loss + F.mse_loss(value, target_return)
                batch_weight += 1

            episode_returns.append(episode.total_return)
            episode_turns.append(episode.turns)

        if batch_weight > 0:
            total_loss = total_policy_loss / batch_weight + value_weight * (total_value_loss / batch_weight)
            total_loss.backward()
            optimizer.step()
            policy_losses.append(float((total_policy_loss / batch_weight).detach().cpu().item()))
            value_losses.append(float((total_value_loss / batch_weight).detach().cpu().item()))

        episodes_remaining -= current_batch

    stats = RLSelfPlayStats(
        episodes=episodes,
        average_return=sum(episode_returns) / max(len(episode_returns), 1),
        average_turns=sum(episode_turns) / max(len(episode_turns), 1),
        policy_loss=sum(policy_losses) / max(len(policy_losses), 1),
        value_loss=sum(value_losses) / max(len(value_losses), 1),
        imitation_loss=imitation_loss,
    )
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        policy.save(checkpoint_path)
    return policy, stats


def train_policy_gradient(*args, **kwargs):
    """Backward-compatible alias for the earlier trainer name."""
    return train_actor_critic(*args, **kwargs)


def warm_start_policy(
    policy: TorchRLPolicy | None = None,
    games: int = 16,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[TorchRLPolicy, ImitationStats]:
    """Pretrain the policy to imitate AdaptiveWeightedBlockingAgent actions."""
    torch_device = resolve_torch_device(device)
    policy = TorchRLPolicy(device=torch_device) if policy is None else policy
    examples = _build_imitation_dataset(games=games)
    optimizer = optim.Adam(policy.network.parameters(), lr=learning_rate)

    epoch_losses: list[float] = []
    for _ in range(epochs):
        random.shuffle(examples)
        running_loss = 0.0
        for example in examples:
            batch = batchify_policy_inputs(example.observation, example.candidates, device=torch_device)
            logits, _ = policy.network(batch)
            target = torch.tensor([example.action_index], device=torch_device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu().item())
        epoch_losses.append(running_loss / max(len(examples), 1))

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        policy.save(checkpoint_path)

    stats = ImitationStats(
        games=games,
        examples=len(examples),
        epochs=epochs,
        loss=sum(epoch_losses) / max(len(epoch_losses), 1),
    )
    return policy, stats


def evaluate_policy(
    checkpoint_path: str | Path,
    games: int = 8,
    opponent: str = "strategic-heuristic",
    device: str | None = None,
) -> RLEvalStats:
    """Evaluate a saved policy against a fixed heuristic opponent lineup."""
    if games <= 0:
        raise ValueError("games must be positive.")

    torch_device = resolve_torch_device(device)
    policy = load_torch_policy(checkpoint_path, device=torch_device)
    rl_agent = RLPolicyAgent(policy=policy, sample_actions=False)
    opponent_factory = _opponent_factory(opponent)

    total_score = 0.0
    total_opponent_score = 0.0
    total_win_credit = 0.0
    total_turns = 0

    for game_index in range(games):
        rl_seat = game_index % 4
        agents = [opponent_factory() for _ in range(4)]
        agents[rl_seat] = rl_agent
        result = play_game(agents=agents, print_boards=False)
        total_turns += result.turn_count
        total_score += result.scores[rl_seat]
        opponent_scores = [score for seat, score in result.scores.items() if seat != rl_seat]
        total_opponent_score += sum(opponent_scores) / len(opponent_scores)
        if rl_seat in result.winners:
            total_win_credit += 1.0 / len(result.winners)

    return RLEvalStats(
        games=games,
        average_score=total_score / games,
        average_opponent_score=total_opponent_score / games,
        win_rate=total_win_credit / games,
        average_turns=total_turns / games,
    )


def _build_imitation_dataset(games: int) -> list[ImitationExample]:
    teacher = AdaptiveWeightedBlockingAgent()
    dataset: list[ImitationExample] = []

    for _ in range(games):
        environment = BlokusRLEnvironment()
        step = environment.reset()
        while not step.done:
            if not step.legal_moves:
                step = environment.step(None)
                continue
            selected_move = teacher.select_move(environment.state, list(step.legal_moves))
            if selected_move is None:
                step = environment.step(None)
                continue
            action_index = list(step.legal_moves).index(selected_move)
            dataset.append(
                ImitationExample(
                    observation=step.observation,
                    candidates=step.candidates,
                    action_index=action_index,
                )
            )
            step = environment.step(selected_move)
    return dataset


def _collect_rollouts(
    policy: TorchRLPolicy,
    episodes: int,
    discount: float,
    workers: int,
) -> list[RolloutEpisode]:
    state_dict = {key: value.detach().cpu() for key, value in policy.network.state_dict().items()}
    if workers <= 1 or episodes <= 1:
        return [_run_rollout_episode(state_dict, seed=index, discount=discount) for index in range(episodes)]

    worker_count = min(workers, episodes)
    episodes_per_worker = [episodes // worker_count for _ in range(worker_count)]
    for index in range(episodes % worker_count):
        episodes_per_worker[index] += 1

    tasks = [
        (state_dict, 10_000 + worker_index, episode_count, discount)
        for worker_index, episode_count in enumerate(episodes_per_worker)
        if episode_count > 0
    ]
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            results = executor.map(_rollout_worker, tasks)
            rollouts: list[RolloutEpisode] = []
            for worker_rollouts in results:
                rollouts.extend(worker_rollouts)
            return rollouts
    except PermissionError:
        return [_run_rollout_episode(state_dict, seed=index, discount=discount) for index in range(episodes)]


def _rollout_worker(task: tuple[dict[str, torch.Tensor], int, int, float]) -> list[RolloutEpisode]:
    state_dict, base_seed, episodes, discount = task
    return [
        _run_rollout_episode(state_dict, seed=base_seed + offset, discount=discount)
        for offset in range(episodes)
    ]


def _run_rollout_episode(
    state_dict: dict[str, torch.Tensor],
    seed: int,
    discount: float,
) -> RolloutEpisode:
    del discount
    rng = random.Random(seed)
    torch.manual_seed(seed)
    policy = TorchRLPolicy(device="cpu")
    policy.network.load_state_dict(state_dict)
    policy.network.eval()

    environment = BlokusRLEnvironment()
    step = environment.reset()
    trajectory: list[RolloutStep] = []
    turns = 0

    while not step.done:
        if not step.legal_moves:
            step = environment.step(None)
            if trajectory:
                last = trajectory[-1]
                trajectory[-1] = RolloutStep(
                    observation=last.observation,
                    candidates=last.candidates,
                    action_index=last.action_index,
                    value_estimate=last.value_estimate,
                    reward=last.reward + step.reward,
                )
            turns += 1
            continue

        batch = batchify_policy_inputs(step.observation, step.candidates, device="cpu")
        with torch.no_grad():
            logits, value = policy.network(batch)
        distribution = torch.distributions.Categorical(logits=logits)
        action_index = int(distribution.sample().item())
        selected_move = step.legal_moves[action_index]
        next_step = environment.step(selected_move)
        trajectory.append(
            RolloutStep(
                observation=step.observation,
                candidates=step.candidates,
                action_index=action_index,
                value_estimate=float(value.item()),
                reward=next_step.reward,
            )
        )
        step = next_step
        turns += 1

    return RolloutEpisode(
        steps=tuple(trajectory),
        total_return=sum(rollout_step.reward for rollout_step in trajectory),
        turns=turns,
    )


def _discounted_returns(
    rewards: list[float],
    discount: float,
    device: str | torch.device,
    normalize: bool,
) -> torch.Tensor:
    returns: list[float] = []
    running_total = 0.0
    for reward in reversed(rewards):
        running_total = reward + discount * running_total
        returns.append(running_total)
    returns.reverse()
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    if normalize and returns_tensor.numel() > 1:
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
    return returns_tensor


def _opponent_factory(name: str):
    factories = {
        "random": RandomAgent,
        "largest": LargestFirstAgent,
        "blocking": BlockingAgent,
        "weighted-blocking": WeightedBlockingAgent,
        "adaptive-weighted-blocking": AdaptiveWeightedBlockingAgent,
        "strategic-heuristic": StrategicHeuristicAgent,
    }
    try:
        return factories[name]
    except KeyError as exc:
        choices = ", ".join(sorted(factories))
        raise ValueError(f"Unknown opponent {name!r}. Expected one of: {choices}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate a PyTorch RL Blokus policy.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run actor-critic self-play training.")
    train_parser.add_argument("--episodes", type=int, default=32, help="Number of self-play episodes.")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    train_parser.add_argument("--discount", type=float, default=0.99, help="Discount factor.")
    train_parser.add_argument("--entropy-weight", type=float, default=0.01, help="Entropy regularization.")
    train_parser.add_argument("--value-weight", type=float, default=0.5, help="Value loss multiplier.")
    train_parser.add_argument("--device", default=None, help="Torch device. Defaults to mps on Apple Silicon.")
    train_parser.add_argument("--workers", type=int, default=1, help="Parallel self-play workers.")
    train_parser.add_argument("--batch-episodes", type=int, default=8, help="Episodes collected per optimizer step.")
    train_parser.add_argument("--warm-start-games", type=int, default=0, help="Teacher self-play games for imitation warm start.")
    train_parser.add_argument("--warm-start-epochs", type=int, default=3, help="Behavior cloning epochs.")
    train_parser.add_argument(
        "--checkpoint",
        default="checkpoints/blokus_rl_policy.pt",
        help="Where to save the trained policy weights.",
    )

    imitate_parser = subparsers.add_parser("imitate", help="Warm start from AdaptiveWeightedBlockingAgent.")
    imitate_parser.add_argument("--games", type=int, default=16, help="Teacher self-play games to sample.")
    imitate_parser.add_argument("--epochs", type=int, default=3, help="Behavior cloning epochs.")
    imitate_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    imitate_parser.add_argument("--device", default=None, help="Torch device. Defaults to mps on Apple Silicon.")
    imitate_parser.add_argument(
        "--checkpoint",
        default="checkpoints/blokus_rl_policy.pt",
        help="Where to save the cloned policy weights.",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved policy against heuristic opponents.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to a saved policy checkpoint.")
    eval_parser.add_argument("--games", type=int, default=8, help="Number of evaluation games.")
    eval_parser.add_argument(
        "--opponent",
        default="strategic-heuristic",
        choices=(
            "random",
            "largest",
            "blocking",
            "weighted-blocking",
            "adaptive-weighted-blocking",
            "strategic-heuristic",
        ),
        help="Opponent policy used for the other three seats.",
    )
    eval_parser.add_argument("--device", default=None, help="Torch device. Defaults to mps on Apple Silicon.")

    args = parser.parse_args()
    if args.command == "train":
        _, stats = train_actor_critic(
            episodes=args.episodes,
            learning_rate=args.learning_rate,
            discount=args.discount,
            entropy_weight=args.entropy_weight,
            value_weight=args.value_weight,
            device=args.device,
            checkpoint_path=args.checkpoint,
            warm_start_games=args.warm_start_games,
            warm_start_epochs=args.warm_start_epochs,
            workers=args.workers,
            batch_episodes=args.batch_episodes,
        )
        print(f"Saved checkpoint: {args.checkpoint}")
        print(f"Device: {resolve_torch_device(args.device)}")
        print(f"Episodes: {stats.episodes}")
        print(f"Average return: {stats.average_return:.4f}")
        print(f"Average turns: {stats.average_turns:.2f}")
        print(f"Policy loss: {stats.policy_loss:.4f}")
        print(f"Value loss: {stats.value_loss:.4f}")
        print(f"Imitation loss: {stats.imitation_loss:.4f}")
        return

    if args.command == "imitate":
        _, stats = warm_start_policy(
            games=args.games,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            checkpoint_path=args.checkpoint,
        )
        print(f"Saved checkpoint: {args.checkpoint}")
        print(f"Device: {resolve_torch_device(args.device)}")
        print(f"Games: {stats.games}")
        print(f"Examples: {stats.examples}")
        print(f"Epochs: {stats.epochs}")
        print(f"Loss: {stats.loss:.4f}")
        return

    stats = evaluate_policy(
        checkpoint_path=args.checkpoint,
        games=args.games,
        opponent=args.opponent,
        device=args.device,
    )
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {resolve_torch_device(args.device)}")
    print(f"Games: {stats.games}")
    print(f"Average RL score: {stats.average_score:.2f}")
    print(f"Average opponent score: {stats.average_opponent_score:.2f}")
    print(f"Win rate: {stats.win_rate:.3f}")
    print(f"Average turns: {stats.average_turns:.2f}")


if __name__ == "__main__":
    main()

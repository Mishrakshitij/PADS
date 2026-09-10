"""Command line entry points; heavyweight dependencies are loaded only on demand."""

import argparse
import json

import numpy as np

from .environment import Environment
from .runner import run_episode, summarize, write_metrics
from .user import AcousticSampler, UserSimulator


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description="PADS bus-dialogue research simulator")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("simulate", "train"):
        command = commands.add_parser(name)
        command.add_argument("--episodes", type=positive_int, default=100 if name == "simulate" else 1000)
        command.add_argument("--max-steps", type=positive_int, default=15)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--clean-prob", type=float, default=0.9)
        command.add_argument("--reward", choices=("task", "politeness"), default="task")
        command.add_argument("--classifier-dir", help="Local four-class DistilBERT checkpoint (not bundled)")
        command.add_argument("--sample-csv", help="Optional simulator acoustic-feature CSV")
        command.add_argument("--output", help="Write per-episode metrics as CSV")
        if name == "simulate":
            command.add_argument("--policy", choices=("slot", "random"), default="slot")
        else:
            command.add_argument("--model-dir", default="outputs/model")
            command.add_argument("--restore", help="Restore a policy checkpoint directory or file prefix")
            command.add_argument("--hidden-dim", type=positive_int, default=32)
            command.add_argument("--learning-rate", type=float, default=0.0001)
            command.add_argument("--discount", type=float, default=0.9)
            command.add_argument("--eval-episodes", type=positive_int, default=100)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.clean_prob <= 1:
        parser.error("--clean-prob must be between 0 and 1")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.reward == "politeness" and not args.classifier_dir:
        parser.error("--reward politeness requires --classifier-dir; the original checkpoint is not bundled")
    if args.reward == "task" and args.classifier_dir:
        parser.error("--classifier-dir requires --reward politeness")
    agent = None
    try:
        scorer = None
        if args.reward == "politeness":
            from .politeness import DistilBertPoliteness
            scorer = DistilBertPoliteness(args.classifier_dir)
        sampler = AcousticSampler(args.sample_csv) if args.sample_csv else None
        env = Environment(UserSimulator(args.clean_prob, args.seed, sampler), scorer)
        rng = np.random.default_rng(args.seed + 1)
        if args.command == "simulate":
            rows = [{"episode": episode, **run_episode(env, max_steps=args.max_steps,
                                                       policy=args.policy, rng=rng)}
                    for episode in range(1, args.episodes + 1)]
            report = {"command": "simulate", "policy": args.policy, "reward": args.reward,
                      "seed": args.seed, **summarize(rows)}
        else:
            from .reinforce import PolicyGradientREINFORCE
            agent = PolicyGradientREINFORCE(hidden_dim=args.hidden_dim,
                                          learning_rate=args.learning_rate,
                                          discount=args.discount, seed=args.seed)
            if args.restore:
                agent.restore(args.restore)
            rows = [{"episode": episode, **run_episode(env, max_steps=args.max_steps,
                                                       agent=agent, training=True)}
                    for episode in range(1, args.episodes + 1)]
            checkpoint = agent.save(args.model_dir)
            eval_env = Environment(UserSimulator(args.clean_prob, args.seed + 1_000_001, sampler), scorer)
            evaluation = [run_episode(eval_env, max_steps=args.max_steps, agent=agent)
                          for _ in range(args.eval_episodes)]
            report = {"command": "train", "reward": args.reward, "seed": args.seed,
                      "training": summarize(rows), "evaluation": summarize(evaluation),
                      "checkpoint": checkpoint}
        if args.output:
            write_metrics(args.output, rows)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    finally:
        if agent is not None:
            agent.close()

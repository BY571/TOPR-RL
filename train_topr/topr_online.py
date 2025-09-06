# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SAC Example.

This is a simple self-contained example of a SAC training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_environment,
    make_replay_buffer,
    make_topr_model,
    make_topr_optimizer,
    make_loss,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("TOPR-Online", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="topr_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger)

    # Create agent
    model, explore_model = make_topr_model(cfg, train_env, eval_env, device)

    # Create TOPR loss
    loss_module = make_loss(cfg.loss, model)

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create off-policy collector
    collector = make_collector(
        cfg, train_env, explore_model, compile_mode=compile_mode
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.replay_buffer.batch_size,
        prb=cfg.replay_buffer.prb,
        r2g_gamma=cfg.replay_buffer.r2g_gamma,
        buffer_size=cfg.replay_buffer.buffer_size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
    )

    # Create optimizers
    optimizer = make_topr_optimizer(cfg.optim, loss_module)

    def update(data):
        optimizer.zero_grad(set_to_none=True)
        # compute losses
        loss_info = loss_module(data)
        actor_loss = loss_info["loss_objective"]
        if cfg.loss.entropy_bonus:
            entropy_loss = loss_info["loss_entropy"]
        else:
            entropy_loss = torch.zeros_like(actor_loss)
            loss_info["loss_entropy"] = entropy_loss

        loss = actor_loss + entropy_loss

        loss.backward()
        optimizer.step()

        return loss_info.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps

    collector_iter = iter(collector)
    total_iter = len(collector)

    for i in range(total_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

        with timeit("collect"):
            tensordict = next(collector_iter)

        # Update weights of the inference policy
        collector.update_policy_weights_()

        current_frames = tensordict.numel()
        pbar.update(current_frames)

        with timeit("rb - extend"):
            # Add to replay buffer
            tensordict = tensordict.reshape(-1)
            replay_buffer.extend(tensordict)

        collected_frames += current_frames

        # Optimization steps
        with timeit("train"):
            if collected_frames >= init_random_frames:
                losses = TensorDict(batch_size=[num_updates])
                for i in range(num_updates):
                    with timeit("rb - sample"):
                        # Sample from replay buffer
                        sampled_tensordict = replay_buffer.sample()

                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss_td = update(sampled_tensordict).clone()
                    losses[i] = loss_td.select(
                        "loss_objective", "loss_entropy"
                    )

                    # Update priority
                    if prb:
                        replay_buffer.update_priority(sampled_tensordict)

        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards
            metrics_to_log["train/episode_length"] = episode_length.sum() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            losses = losses.mean()
            metrics_to_log["train/objective_loss"] = losses.get("loss_objective")
            metrics_to_log["train/entropy_loss"] = losses.get("loss_entropy")
            metrics_to_log["train/negatives"] = losses.get("negatives")

        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval"):
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
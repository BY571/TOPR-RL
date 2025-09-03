"""SAC Example.

This is a simple self-contained example of a SAC eval script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import hydra
import numpy as np
import torch
import torch.cuda
from tqdm import tqdm
from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utils import (
    make_environment,
    make_replay_buffer,
    make_sac_agent,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    total_eval_steps = cfg.eval_steps
    max_rollout_steps = cfg.env.max_episode_steps
    weight_path = cfg.weight_path
    policy_type = cfg.policy_type

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg)

    # Create agent
    model, _ = make_sac_agent(cfg, train_env, eval_env, device)

    # Load Model weights -- for untrained policy you just skip this step
    model.load_state_dict(torch.load(weight_path))

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
    )

    # Run Evaluation
    total_collected = 0
    pbar = tqdm(total=total_eval_steps, desc="Evaluating", unit="steps")
    for i in range(1000):
        with set_exploration_type(
            ExplorationType.DETERMINISTIC
        ), torch.no_grad(), timeit("eval"):
            eval_rollout = eval_env.rollout(
                max_rollout_steps,
                model[0],
                auto_cast_to_device=True,
                break_when_any_done=True, # we want to continue sample until we reach the required steps
            )

        episode_end = (
            eval_rollout["next", "done"]
            if eval_rollout["next", "done"].any()
            else eval_rollout["next", "truncated"]
        )
        episode_rewards = eval_rollout["next", "episode_reward"][episode_end]
        episode_length = eval_rollout["next", "step_count"][episode_end]
        print("*** Evaluation Stats: ***")
        print(f"Episode rewards: {episode_rewards.mean()}")
        print(f"Episode rewards std: {episode_rewards.std()}")
        print(f"Episode count: {len(episode_rewards)}")
        print(f"Episode length: {episode_length.sum() / len(episode_length)}")
        # could do some preprocessing here
        eval_rollout = eval_rollout.cpu().reshape(-1)
        steps_collected = eval_rollout.batch_size[0]
        total_collected += steps_collected
        pbar.update(steps_collected)
        pbar.set_postfix({
            'collected': f'{total_collected}/{total_eval_steps}'
        })
        replay_buffer.extend(eval_rollout)
        if total_collected >= total_eval_steps:
            break

    pbar.close()
    replay_buffer.dumps(f"./replay_buffer_{policy_type}.pt")

    

if __name__ == "__main__":
    main()
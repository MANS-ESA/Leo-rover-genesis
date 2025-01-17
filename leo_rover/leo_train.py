import argparse
import os
import pickle
import shutil
import numpy

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner
from leo_env import LeoRoverEnv  

# run : python leo_train.py --exp_name leo-rover-run --max_iterations 2000
def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": 20, 
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 50,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }
    return train_cfg_dict



def get_cfgs():
    env_cfg = {
        "num_actions": 2,
        # base pose
        "base_init_pos": [1.0, .0, 1.0],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 25.0,
        "at_target_threshold": 0.1,
        "clip_actions": 1.0,
        # visualization
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "num_obs": 5,
        "obs_scales": {
            "rel_pos": 1 / 3.0,
            "lin_vel": 1 / 3.0,
            "ang_vel": 1 / 3.14159,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "target": 10.0,
            "smooth": -1e-4,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "pos_x_range": [-2.0, 2.0],
        "pos_y_range": [0.1, 0.15],
        "pos_z_range": [-2.0, 2.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="leo-rover-run")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=2)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--urdf_path", type=str, default=r"/Users/julienlegrand/Documents/IG2I/ESA/Leo-rover-genesis/URDF/leo_sim.urdf")
    parser.add_argument("--device", type=str, default=r"mps")
    args = parser.parse_args()

    gs.init(logging_level="error") #info si voir FPS

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True

    env = LeoRoverEnv(
        num_envs=args.num_envs,
        urdf_path=args.urdf_path,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        device=args.device
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Start learning
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()

import argparse
import os
import pickle
import shutil

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
            "activation": "elu",
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="leo-rover-run")
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--urdf_path", type=str, default=r"C:\Users\flori\Downloads\my_leo_robot\leo_sim.urdf")
    args = parser.parse_args()

    gs.init(logging_level="info")

    # Création d’un dossier de logs
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Création de l’environnement
    env = LeoRoverEnv(
        num_envs=args.num_envs,
        urdf_path=args.urdf_path,
        show_viewer=False,
        device="cuda:0"
    )

    # Configuration d’entraînement
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # Création du runner OnPolicy (PPO, etc.)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    # Sauvegarde de la config
    pickle.dump(
        {
            "env_cfg": None,
            "train_cfg": train_cfg,
            "urdf_path": args.urdf_path,
        },
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Lancement de l’apprentissage
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()

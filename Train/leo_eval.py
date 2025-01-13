import argparse
import os
import pickle
import torch
from rsl_rl.runners import OnPolicyRunner
from leo_env import LeoRoverEnv 

import genesis as gs

#run : python leo_eval.py -e leo-rover-run --ckpt 50 --show_viewer

def main():
    # Arguments pour l'évaluation
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="leo-rover-run")
    parser.add_argument("--ckpt", type=int, default=50) 
    parser.add_argument("--show_viewer", action="store_true", default=False, help="Affiche la vue 3D en temps réel.")
    args = parser.parse_args()

    # Initialisation de Genesis
    gs.init(logging_level="info")

    # Dossier de logs contenant le modèle et la configuration
    log_dir = f"logs/{args.exp_name}"
    config_path = os.path.join(log_dir, "cfgs.pkl")

    # Charger la configuration de l'entraînement
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration non trouvée : {config_path}")
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(config_path, "rb"))

    # Réinitialiser les récompenses pour ne pas interférer avec l'évaluation
    reward_cfg["reward_scales"] = {}

    # Création de l'environnement pour l'évaluation
    env = LeoRoverEnv(
        num_envs=1,  # Une seule instance pour l'évaluation
        urdf_path=r"C:\Users\flori\Downloads\my_leo_robot\leo_sim.urdf",
        show_viewer=args.show_viewer,  # Affiche ou non la visualisation
        device="cuda:0",
    )

    # Chargement du modèle entraîné
    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    model_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint non trouvé : {model_path}")
    runner.load(model_path)

    # Récupérer la politique pour l'inférence
    policy = runner.get_inference_policy(device="cuda:0")

    # Démarrage de l'évaluation
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            # Calcul des actions et passage à l'étape suivante
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

            # Si l'épisode se termine, réinitialiser
            if dones[0].item():
                obs, _ = env.reset()


if __name__ == "__main__":
    main()

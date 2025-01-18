import argparse
import time
import torch
import numpy as np
import genesis as gs
from pynput import keyboard


class RoverController:
    def __init__(self):
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.max_linear_velocity = 0.4
        self.min_linear_velocity = -0.4
        self.max_angular_velocity = 1.0
        self.min_angular_velocity = -1.0

        self.running = True
        self.pressed_keys = set()

    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            self.pressed_keys.add(key)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key)
        except KeyError:
            pass

    def compute_action(self):
        """ Calcule les actions en fonction des touches pressées """
        linear_velocity = 0.0
        angular_velocity = 0.0

        if 'z' in self.pressed_keys:
            linear_velocity = self.max_linear_velocity
        if 's' in self.pressed_keys:
            linear_velocity = self.min_linear_velocity
        if 'q' in self.pressed_keys:
            angular_velocity = self.max_angular_velocity
        if 'd' in self.pressed_keys:
            angular_velocity = self.min_angular_velocity

        return np.array([[linear_velocity, angular_velocity]], dtype=np.float32)


def update_camera(scene, rover):
    """ Met à jour la caméra pour suivre le rover """
    if not scene.viewer:
        return

    rover_pos = rover.get_pos()

    # Définition de la position de la caméra par rapport au rover
    offset_x = 0.0
    offset_y = -2.0  # 2 unités derrière
    offset_z = 1.0   # 1 unité au-dessus

    camera_pos = (
        float(rover_pos[0] + offset_x),
        float(rover_pos[1] + offset_y),
        float(rover_pos[2] + offset_z),
    )

    scene.viewer.set_camera_pose(pos=camera_pos, lookat=tuple(float(x) for x in rover_pos))


def run_sim(scene, rover, controller):
    """ Boucle principale de simulation """
    while controller.running:
        try:
            # Récupérer les actions
            actions = controller.compute_action()

            # Appliquer les commandes au rover
            actions_tensor = torch.tensor(actions, device="cpu")
            linear_vel = actions_tensor[:, 0:1]  # Vitesse linéaire
            angular_vel = actions_tensor[:, 1:2]  # Vitesse angulaire

            wheel_base = 0.359  # Distance entre les roues
            left_wheel_vel = linear_vel - (angular_vel * wheel_base / 2)
            right_wheel_vel = linear_vel + (angular_vel * wheel_base / 2)

            left_wheel_vel_deg = left_wheel_vel * (180 / np.pi)
            right_wheel_vel_deg = right_wheel_vel * (180 / np.pi)

            # Appliquer les vitesses aux roues du rover
            rover.control_dofs_velocity(
                torch.cat([left_wheel_vel, right_wheel_vel, left_wheel_vel, right_wheel_vel], dim=-1),
                rover.motor_dofs,
            )

            # Avancer la simulation
            scene.step()

            # Mettre à jour la caméra
            update_camera(scene, rover)

            # Limiter la boucle à ~60 FPS
            time.sleep(1 / 60)

        except Exception as e:
            print(f"Erreur dans la boucle de simulation : {e}")

    if scene.viewer:
        scene.viewer.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Activer la visualisation")
    parser.add_argument("-f", "--follow", action="store_true", default=True, help="Activer la caméra qui suit")
    args = parser.parse_args()

    # Initialisation de Genesis
    gs.init(backend=gs.cuda)

    # Paramètres de la caméra
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0.0, -2.0, 1.0),  # Derrière le rover
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=45,
        max_FPS=60,
    )

    # Création de la scène
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    # Ajouter un sol
    plane = scene.add_entity(gs.morphs.Plane())

    # Ajouter le rover
    rover = scene.add_entity(
        gs.morphs.URDF(
            file="../URDF/leo_sim.urdf",
            pos=(0.0, 0.0, 0.1),
        )
    )

    # Construire la scène
    # Récupérer les index des moteurs du rover
    dof_names = ["wheel_FL_joint", "wheel_FR_joint", "wheel_RL_joint", "wheel_RR_joint"]
    motor_dofs = [rover.get_joint(name).dof_idx_local for name in dof_names]

    rover.motor_dofs = motor_dofs  # Stocker les DOFs moteurs pour le contrôler

    # Contrôleur du rover
    controller = RoverController()

    # Démarrer l'écoute clavier
    listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    listener.start()

    if args.follow:
        # Ajoute une caméra qui suit le rover
        camera = scene.add_camera(
            res=(640, 480),
            pos=(0.0, -2.0, 1.0),
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            GUI=True,
        )
        #scene.set_camera_follow_entity(camera, rover, height=1.0, smoothing=0.1)

    # Lancer la simulation
    scene.build()

    run_sim(scene, rover, controller)
    listener.stop()


if __name__ == "__main__":
    main()

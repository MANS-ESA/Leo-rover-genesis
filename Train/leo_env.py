import math
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class LeoRoverEnv:
    """
    Exemple d'environnement pour un rover 4 roues avec Genesys AI
    qui va vers une coordonnée cible (target_pos).
    """

    def __init__(
        self,
        num_envs=4096,
        urdf_path=r"C:\Users\flori\Downloads\my_leo_robot\leo_sim.urdf",
        target_pos=(2.0, 2.0),   # Coordonnée cible XY (exemple)
        show_viewer=False,
        device="cuda:0",
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs

        # ---------------------------------------------------
        # Paramètres de base : vitesses, fréquences, etc.
        # ---------------------------------------------------
        self.dt = 0.02            # 50 Hz
        self.max_episode_length = 500  # nombre de steps maxi
        self.num_actions = 4

        self.num_obs = 10
        self.num_privileged_obs = None

        self.target_pos = torch.tensor(target_pos, dtype=torch.float, device=self.device).repeat(num_envs, 1)

        # ---------------------------------------------------
        # Création de la scène Genesis
        # ---------------------------------------------------
        sim_options = gs.options.SimOptions(dt=self.dt, substeps=2)
        viewer_options = gs.options.ViewerOptions(
            max_FPS=int(1 / self.dt),
            camera_pos=(2.0, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=60,
        )
        vis_options = gs.options.VisOptions(n_rendered_envs=1)
        rigid_options = gs.options.RigidOptions(
            dt=self.dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True
        )

        self.scene = gs.Scene(
            sim_options=sim_options,
            viewer_options=viewer_options,
            vis_options=vis_options,
            rigid_options=rigid_options,
            show_viewer=show_viewer,
        )

        # ---------------------------------------------------
        # Ajout d’un plan de base + rover
        # ---------------------------------------------------
        # Le plan
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # Le rover (Leo)
        # Remplacez par le chemin correct vers votre URDF
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=[0.0, 0.0, 0.1],  # Position initiale au-dessus du sol
                quat=[1.0, 0.0, 0.0, 0.0],
            )
        )

        # On construit la scène pour num_envs environnements
        self.scene.build(n_envs=self.num_envs)

        # Récupération des index de DOF pour chaque roue
        # NOTE : adaptez aux noms réels de vos joints dans l'URDF
        # ci-dessous, j'utilise par exemple "wheel_FL_joint" etc.
        self.dof_names = [
            "wheel_FL_joint",
            "wheel_FR_joint",
            "wheel_RL_joint",
            "wheel_RR_joint",
        ]
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]

        # Paramètres de base pour le control (PD ou direct velocity)
        self.kp = 0.0    # si on contrôle en vitesse
        self.kd = 0.0
        self.robot.set_dofs_kp([self.kp] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.kd] * self.num_actions, self.motor_dofs)

        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.long, device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # Vitesse lin/ang de la base
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        # Pose du rover
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device)

    def step(self, actions):
        """
        Avance la simulation d'un pas en appliquant l'action.
        actions shape: (num_envs, 4) => vitesses pour chaque roue
        """
        # Appliquer directement les actions comme des cibles de vitesse
        # sur les joints. On doit convertir [rad/s] => [deg/s] si necessary.
        # Ici, on suppose que l'action est déjà dans le bon ordre de grandeur.
        target_velocity_deg = actions * 180.0 / math.pi  # exemple
        self.robot.control_dofs_velocity(target_velocity_deg, self.motor_dofs)

        # Avancer la simulation
        self.scene.step()

        # Mettre à jour les buffers
        self.episode_length_buf += 1

        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        inv_quat_ = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_quat_)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_quat_)
        vel_joints = self.robot.get_dofs_velocity(self.motor_dofs)
        self.dof_vel[:] = vel_joints

        # Calcul des récompenses
        self._compute_reward()

        # Vérifier terminaisons
        self.reset_buf = self._check_termination()

        # Reset de ceux qui terminent
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Observations
        self._compute_observations()

        return self.obs_buf, None, self.rew_buf, self.reset_buf, {}

    def reset(self):
        """
        Reset complet de tous les environnements
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        return self.obs_buf, None

    def reset_idx(self, env_ids):
        """
        Reset sélectif de certains environnements
        """
        if len(env_ids) == 0:
            return

        # Remettre le rover au centre (par ex)
        reset_pos = torch.zeros((len(env_ids), 3), device=self.device)
        reset_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)

        self.robot.set_pos(reset_pos, envs_idx=env_ids)
        self.robot.set_quat(reset_quat, envs_idx=env_ids)
        self.robot.zero_all_dofs_velocity(envs_idx=env_ids)

        # On peut randomiser la cible ou la laisser fixe
        # Ex: target_pos X et Y dans [1, 3]
        self.target_pos[env_ids, 0] = gs_rand_float(1.0, 3.0, (len(env_ids),), self.device)
        self.target_pos[env_ids, 1] = gs_rand_float(1.0, 3.0, (len(env_ids),), self.device)

        # Remise à zéro des compteurs
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # Recalcule la première obs
        self._compute_observations()

    def _compute_observations(self):

        rover_xy = self.base_pos[:, 0:2]  # [N, 2]
        
        to_target = (self.target_pos - rover_xy)
        dist_to_target = torch.norm(to_target, dim=-1, keepdim=True)  
        dir_to_target = to_target / (dist_to_target + 1e-6)           
        lin_vel_xy = self.base_lin_vel[:, 0:2] 

       
        ang_vel_z = self.base_ang_vel[:, 2:3]   

   
        self.obs_buf = torch.cat(
            [
                lin_vel_xy,   # 2
                ang_vel_z,    # 1
                dir_to_target,   # 2
                dist_to_target,  # 1
                self.dof_vel,    # 4
            ],
            dim=-1
        )

    def _compute_reward(self):
        rover_xy = self.base_pos[:, 0:2]
        dist_to_target = torch.norm(self.target_pos - rover_xy, dim=-1)
        rew_dist = -dist_to_target
        rew_reach = (dist_to_target < 0.2).float() * 5.0  

        speed_penalty = -0.01 * torch.sum(self.dof_vel**2, dim=-1)
        self.rew_buf[:] = rew_dist + rew_reach + speed_penalty

    def _check_termination(self):
        done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        done = done | (self.episode_length_buf >= self.max_episode_length)

        rover_xy = self.base_pos[:, 0:2]
        dist_to_target = torch.norm(self.target_pos - rover_xy, dim=-1)
        done = done | (dist_to_target < 0.1)

        return done

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

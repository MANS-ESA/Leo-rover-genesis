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
        urdf_path=r"../URDF/leo_sim.urdf",
        target_pos=(2.0, 2.0),   # Coordonnée cible XY (exemple)
        show_viewer=False,
        device="mps",
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs

        # ---------------------------------------------------
        # Paramètres de base : vitesses, fréquences, etc.
        # ---------------------------------------------------
        self.dt = 0.02            # 50 Hz
        self.max_episode_length = 2000  # nombre de steps maxi


        self.num_obs = 6
        self.num_privileged_obs = None
        self.target_pos = torch.tensor([target_pos], dtype=torch.float, device=self.device).repeat(num_envs, 1)
        self.ball_pos = torch.tensor([(2.0, 2.0, 2.0)], dtype=torch.float, device=self.device).repeat(num_envs, 1)
        self.num_actions = 2 #torch.zeros((num_envs,self.num_obs),device = self.device, dtype = gs.tc_float)


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

        self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=True,
                    collision=False,
                )
        )

        # ---------------------------------------------------
        # Ajout d’un plan de base + rover
        # ---------------------------------------------------
        # Le plan
        self.scene.add_entity(gs.morphs.Plane())

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
        self.kp = 1_000_000    # si on contrôle en vitesse
        self.kd = 1.0
        self.robot.set_dofs_kp([self.kp] * len(self.motor_dofs), self.motor_dofs)
        self.robot.set_dofs_kv([self.kd] * len(self.motor_dofs), self.motor_dofs)

        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.long, device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)  # Linear and angular commands



        # Vitesse lin/ang de la base
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)

        # Pose du rover
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_dist_to_target = torch.norm(self.target_pos - self.base_pos[:, 0:2], dim=-1)
        #feedback 
        self.last_actions = torch.zeros_like(self.actions)

        #  position du over moment du spawn
        #self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, len(self.motor_dofs)), device=self.device)

        #self.base_quat = torch.zeros((self.num_envs, 4), device=self.device)

    def step(self, actions):
        """
        Advances the simulation by one step, applying linear and angular commands.
        actions: [N, 2] -> linear and angular velocities for each environment
        """
        # Clip actions to valid ranges
        self.actions = torch.clip(actions, -4, 4)

        self.episode_length_buf += 1
        # Convert linear and angular commands to wheel velocities (example logic)
        linear_vel = self.actions[:, 0:1]  # Linear velocity [N, 1]
        angular_vel = self.actions[:, 1:2]  # Angular velocity [N, 1]

        # Assume a simple differential drive model for the rover
        wheel_base = 0.359 
        left_wheel_vel = linear_vel - (angular_vel * wheel_base / 2)
        right_wheel_vel = linear_vel + (angular_vel * wheel_base / 2)
        wheel_velocities = torch.cat([left_wheel_vel, right_wheel_vel, left_wheel_vel, right_wheel_vel], dim=-1)  # Shape: [N, 4]

        # Control the robot's wheels
        self.robot.control_dofs_velocity(wheel_velocities, self.motor_dofs)

        # Advance the simulation
        self.scene.step()

        # Update observations, rewards, and termination conditions
        self._compute_observations()
        self._compute_reward()
        self.reset_buf = self._check_termination()

        # Reset environments that have finished
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

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
        self.ball_pos[env_ids, 0] = self.target_pos[env_ids, 0]
        self.ball_pos[env_ids, 1] = self.target_pos[env_ids, 1]
        self.ball_pos[env_ids, 2] = gs_rand_float(1.0, 3.0, (len(env_ids),), self.device)
        # Remise à zéro des compteurs
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.last_actions[env_ids] = torch.zeros(len(env_ids), self.num_actions, device=self.device)
        
        self.target.set_pos(self.ball_pos[env_ids], zero_velocity=True, envs_idx=env_ids)

        self._compute_observations()

    def _compute_observations(self):

        rover_xy = self.base_pos[:, 0:2]  # [N, 2]
        #to_target = (self.target_pos - rover_xy)

        #dist_to_target = torch.norm(to_target, dim=-1, keepdim=True)  
        #dir_to_target = to_target / (dist_to_target + 1e-6)           
        lin_vel_xy = self.base_lin_vel[:, 0:1] 
        ang_vel_z = self.base_ang_vel[:, 2:3]   

   
        self.obs_buf = torch.cat(
            [
                rover_xy,   # rover xy
                self.target_pos,    # target position
                self.last_actions, #son petit feedback
            ],
            dim=-1
        )

        self.last_actions[:] = self.actions[:]


    def _compute_reward(self):
        rover_xy = self.base_pos[:, 0:2]
        dist_to_target = torch.norm(self.target_pos - rover_xy, dim=-1)
        rew_dist = -dist_to_target
        rew_reach = (dist_to_target < 0.2).float() * 5.0

        delta = dist_to_target - self.last_dist_to_target
        self.last_dist_to_target = dist_to_target

        self.rew_buf[:] = rew_dist + delta

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

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
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        urdf_path=r"../URDF/leo_sim.urdf",
        show_viewer=False,
        device="mps",
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs

        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.dt = 0.02            # 50 Hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]


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

        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="../meshes/sphere.obj",
                    scale=0.3,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # Add Leo Rover
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.robot = self.scene.add_entity(gs.morphs.URDF(
            file=urdf_path,
            pos=(0.0, 0.0, 0.0),
        ))

        # build scene
        self.scene.build(n_envs=self.num_envs)

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

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Initialize buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=torch.long, device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_base_pos = torch.zeros_like(self.base_pos)

        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["pos_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["pos_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["pos_z_range"], (len(envs_idx),), self.device)
        if self.target is not None:
            self.target.set_pos(self.commands[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
        )
        return at_target

    def step(self, actions):

        self.actions[:,0:1] = torch.clip(actions[:,0:1], -self.env_cfg["clip_actions_lin"], self.env_cfg["clip_actions_lin"])
        self.actions[:,1:2] = torch.clip(actions[:,1:2], -self.env_cfg["clip_actions_ang"], self.env_cfg["clip_actions_ang"])

        linear_vel = self.actions[:, 0:1]  # Linear velocity [m/s, 1]
        angular_vel = self.actions[:, 1:2]  # Angular velocity [rad/s, 1]



        wheel_base = 0.359
        wheel_radius = 0.033  # Radius of the wheels (m)
        #left_wheel_vel = linear_vel - (angular_vel * wheel_base / 2)
        #right_wheel_vel = linear_vel + (angular_vel * wheel_base / 2)
        left_wheel_vel = (linear_vel -  2.0* (angular_vel * wheel_base / 2)) / wheel_radius
        right_wheel_vel = (linear_vel + 2.0* (angular_vel * wheel_base / 2)) / wheel_radius
        #left_wheel_vel_deg = left_wheel_vel * (180 / torch.pi)
        #right_wheel_vel_deg = right_wheel_vel * (180 / torch.pi)
        self.robot.control_dofs_velocity(
            torch.cat([left_wheel_vel, right_wheel_vel, left_wheel_vel, right_wheel_vel], dim=-1)
            ,self.motor_dofs
        )

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.last_base_pos[:] = self.base_pos[:]
        self.base_pos[:] = self.robot.get_pos()
        self.rel_pos = self.commands - self.base_pos
        #print(f"rel pose : {self.rel_pos}")
        self.last_rel_pos = self.commands - self.last_base_pos

        # resample commands
        #envs_idx = self._at_target()

        #add some rew
        #self._resample_commands(envs_idx)

        self.crash_condition = (
            (torch.abs(self.rel_pos[:, 0]) > self.env_cfg["termination_if_x_greater_than"])
            | (torch.abs(self.rel_pos[:, 2]) > self.env_cfg["termination_if_z_greater_than"])
        )

        self.target_touched = self._at_target()

        # check termination and reset
        self.reset_buf = (self.episode_length_buf > self.max_episode_length) | self.crash_condition | self.target_touched

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        self.rew_buf[:] = 0.0
        # compute reward
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.commands,
                self.base_pos,
                self.last_actions,
            ],
            axis=-1
        )
        self.last_actions[:] = self.actions[:]


        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras


    def reset(self):
        """
        Reset complet de tous les environnements
        """
        self.reset_buf[:] = True
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        return self.obs_buf, None

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        # reset base
        self.base_pos[env_ids] = self.base_init_pos
        self.last_base_pos[env_ids] = self.base_init_pos
        self.rel_pos = self.commands - self.base_pos
        self.last_rel_pos = self.commands - self.last_base_pos
        self.robot.set_pos(self.base_pos[env_ids], zero_velocity=True, envs_idx=env_ids)
        self.base_lin_vel[env_ids] = 0
        self.robot.zero_all_dofs_velocity(env_ids)


        # reset buffers
        self.last_actions[env_ids] = torch.zeros_like(self.last_actions[env_ids])
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = True

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][env_ids] = 0.0

        self._resample_commands(env_ids)


    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_pos), dim=1) - torch.sum(torch.square(self.rel_pos), dim=1)
        return target_rew
    
    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_duration(self):
        return self.episode_length_buf.float()
    
    def _reward_crash(self):
        crash_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        crash_rew[self.crash_condition] = 1
        return crash_rew
    
    def _reward_target_touched(self):
        target_touched_rew = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        target_touched_rew[self.target_touched] = 1
        return target_touched_rew

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None
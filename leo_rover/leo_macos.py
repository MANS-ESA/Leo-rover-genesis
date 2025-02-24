import argparse

import torch
import numpy as np
import genesis as gs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(backend=gs.metal)

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            res = (1500, 1000),
            camera_lookat=(-1.0, 1.0, 0),
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            dt=0.02,
            gravity=(0.0, 0.0, -10.0),
        ),
    )

    cam = scene.add_camera(
        res    = (1280, 960),
        pos    = (3.5, 0.0, 2.5),
        lookat = (0, 0, 0.5),
        fov    = 30,
        GUI    = True
    )

    ########################## entities ##########################
    plane = scene.add_entity(gs.morphs.Plane())
    r0 = scene.add_entity(
        gs.morphs.URDF(
            file="/Users/julienlegrand/Documents/IG2I/ESA/Leo-rover-genesis/URDF/leo_sim.urdf",
        ),
    )

    ########################## build ##########################
    scene.build()

    '''jnt_names = [
        'wheel_FL_joint',
        'wheel_RL_joint',
        'wheel_FR_joint',
        'wheel_RR_joint',
    ]

    dofs_idx = [r0.get_joint(name).dof_idx_local for name in jnt_names]

    # set positional gains
    r0.set_dofs_kp(
        kp             = np.array([1_000_000, 1_000_000, 1_000_000, 1_000_000]),
        dofs_idx_local = dofs_idx,
    )
    # set velocity gains
    r0.set_dofs_kv(
        kv             = np.array([1.0, 1.0, 1.0, 1.0]),
        dofs_idx_local = dofs_idx,
    )'''


    linear_vel = 0.4
    angular_vel = 0.8
    robot_angular_velocity_multiplier = 1.76
    angular_vel = angular_vel * robot_angular_velocity_multiplier
    wheel_encoder_resolution = 1

    wheel_base = 0.358
    wheel_radius = 0.0625
    left_wheel_vel = linear_vel - (angular_vel * wheel_base / 2)
    wheel_L_ang_vel = left_wheel_vel / wheel_radius

    right_wheel_vel = linear_vel + (angular_vel * wheel_base / 2)
    wheel_R_ang_vel = right_wheel_vel / wheel_radius

    left_wheel_speed = (wheel_L_ang_vel / (2 * torch.pi)) * wheel_encoder_resolution
    right_wheel_speed = (wheel_R_ang_vel / (2 * torch.pi)) * wheel_encoder_resolution

    ''''r0.control_dofs_velocity(
                np.array([left_wheel_speed, left_wheel_speed, right_wheel_speed, right_wheel_speed]),
                dofs_idx,
            )'''

    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, r0))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, robot):
    from time import time
    robot.set_pos([10.0, 10.0, 1.0], zero_velocity=True)

    t_prev = time()
    i = 0
    while True:
        i += 1
        scene.step()
        print(f"Robot position : {robot.get_pos()}")
        t_now = time()
        print(1 / (t_now - t_prev), "FPS")
        t_prev = t_now
        if i > 5000:
            break

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()

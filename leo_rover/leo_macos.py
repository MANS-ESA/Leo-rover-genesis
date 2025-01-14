import argparse

import torch

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
            dt=0.01,
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
            file="./leo_sim.urdf",
            pos=(0, 0, 0)
        ),
    )

    ########################## build ##########################
    scene.build()
    gs.tools.run_in_another_thread(fn=run_sim, args=(scene, args.vis, cam))
    if args.vis:
        scene.viewer.start()


def run_sim(scene, enable_vis, cam):
    from time import time

    t_prev = time()
    i = 0
    while True:
        i += 1
        scene.step()
        t_now = time()
        print(1 / (t_now - t_prev), "FPS")
        t_prev = t_now
        if i > 1000:
            break

    if enable_vis:
        scene.viewer.stop()


if __name__ == "__main__":
    main()

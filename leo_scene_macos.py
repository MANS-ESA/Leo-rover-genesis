import argparse

import torch

import genesis as gs


def main():
    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            res = (1500, 1000),
            camera_lookat=(-1.0, 1.0, 0),
        ),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0.0, 0.0, -10.0),
        ),
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
    run_sim(scene)


def run_sim(scene):
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


if __name__ == "__main__":
    main()

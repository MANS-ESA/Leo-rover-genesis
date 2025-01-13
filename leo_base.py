import genesis as gs

gs.init(backend=gs.metal)

scene = gs.Scene()

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    # gs.morphs.URDF(
    #     file='urdf/panda_bullet/panda.urdf',
    #     fixed=True,
    # ),
    gs.morphs.URDF(file="./leo_sim.urdf"),
)

def run_sim(scene): 
    for i in range(1000):
        scene.step()

scene.build()
gs.tools.run_in_another_thread(fn=run_sim, args=(scene, ))
scene.viewer.start()
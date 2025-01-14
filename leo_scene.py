import genesis as gs
gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.URDF(file='./URDF/leo_sim.urdf'),
)

scene.build()

for i in range(1000):
    scene.step()
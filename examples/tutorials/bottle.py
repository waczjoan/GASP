import genesis as gs
import torch

########################## init ##########################
gs.init()

########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, -1.0),
        upper_bound=(1.0, 1.0, 1),
    ),
    vis_options=gs.options.VisOptions(
        visualize_mpm_boundary=True,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(960, 640),
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(pos=(-1, -1, -1)),
)

SCALE=0.4
mesh = scene.add_entity(
    material=gs.materials.MPM.Elastic(sampler="random"),#(E=1e5, nu=0.1, rho=1000),
    morph=gs.morphs.Mesh(file="scale_1.obj", convexify=False, decompose_nonconvex=False, scale=SCALE)
)

scene.build()

########################## build ##########################


horizon = 1000
for i in range(horizon):
    scene.step()
    print(torch.tensor(mesh.get_state().pos[mesh.particle_start:mesh.particle_end]).shape)

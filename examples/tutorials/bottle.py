import genesis as gs
import torch

########################## init ##########################
gs.init()

########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=1e-2,
        substeps=100,
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


import os
materials = {
    "base": gs.materials.MPM.Base,
    "elastic": gs.materials.MPM.Elastic,
    "elastoplastic": gs.materials.MPM.ElastoPlastic,
    "liquid": gs.materials.MPM.Liquid,
    "muscle": gs.materials.MPM.Muscle,
    "sand": gs.materials.MPM.Sand,
    "snow": gs.materials.MPM.Snow,
}

MATERIAL="sand"
MODEL_PATH="output\\bottle"
OBJ_PATH=os.path.join(MODEL_PATH, "pseudomesh_info\\ours_30000\\scale_1.obj")
SAVE_PATH=os.path.join(MODEL_PATH, "genesis_triangles", MATERIAL)
os.makedirs(SAVE_PATH, exist_ok=True)
SCALE=0.4
mesh = scene.add_entity(
    material=materials[MATERIAL](sampler="random"),#(E=1e5, nu=0.1, rho=1000),
    morph=gs.morphs.Mesh(file=OBJ_PATH, convexify=False, decompose_nonconvex=False, scale=SCALE)
)

scene.build()

########################## build ##########################


def calc_scales(pts):
    if isinstance(pts, list):
        pts = torch.concatenate(pts)
    x = pts
    scales = x.reshape((-1, 3, 3))
    scales = scales - scales[:, 0, :].unsqueeze(-2)
    scales = torch.linalg.norm(scales, axis=-1)
    return scales.reshape(-1, 1)

def clamp_pos(x, init_scales, threshold=1.5):
    scales = calc_scales(x).expand(-1, 3)
    return torch.where(scales > threshold * init_scales, x + threshold * init_scales, x)


init_scales = calc_scales(mesh.get_state().pos[mesh.particle_start:mesh.particle_end]).expand(-1, 3)

horizon = 800
for i in range(horizon + 1):
    scene.step()
    if i % 4 == 0:
        mesh.get_state().pos[mesh.particle_start:mesh.particle_end] = clamp_pos(mesh.get_state().pos[mesh.particle_start:mesh.particle_end], init_scales)
        torch.save(
            torch.tensor(mesh.get_state().pos[mesh.particle_start:mesh.particle_end].clone().detach().to("cpu")).reshape(-1, 3, 3),
            os.path.join(SAVE_PATH, f"{i:05d}.pt")
        )

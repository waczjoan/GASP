import genesis as gs
import torch
import argparse

########################## init ##########################
gs.init()
materials = {
    "base": gs.materials.MPM.Base,
    "elastic": gs.materials.MPM.Elastic,
    "elastoplastic": gs.materials.MPM.ElastoPlastic,
    "liquid": gs.materials.MPM.Liquid,
    "muscle": gs.materials.MPM.Muscle,
    "sand": gs.materials.MPM.Sand,
    "snow": gs.materials.MPM.Snow,
}

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


######################### get args #########################
parser = argparse.ArgumentParser()
parser.add_argument("--obj_path")
parser.add_argument("--model_path")
parser.add_argument("--save_path", default=None)
parser.add_argument("--material", choices=materials.keys())
args = parser.parse_args()
material = args.material
model_path = args.model_path
obj_path = args.obj_path or os.path.join(model_path, "pseudomesh_info\\ours_30000\\scale_1.obj")
save_path = args.save_path or os.path.join(model_path, "genesis_triangles", material)

########################## entities ##########################
plane = scene.add_entity(
    morph=gs.morphs.Plane(pos=(-1, -1, -1)),
)


import os

os.makedirs(save_path, exist_ok=True)
SCALE=0.5
mesh = scene.add_entity(
    material=materials[material](sampler="random"),#(E=1e5, nu=0.1, rho=1000),
    morph=gs.morphs.Mesh(file=obj_path, convexify=False, decompose_nonconvex=False, scale=SCALE)
)

scene.build()

########################## build ##########################


# def calc_scales(pts):
#     if isinstance(pts, list):
#         pts = torch.concatenate(pts)
#     x = pts
#     scales = x.reshape((-1, 3, 3))
#     scales = scales - scales[:, 0, :].unsqueeze(-2)
#     scales = torch.linalg.norm(scales, axis=-1)
#     return scales.reshape(-1, 1)

# def clamp_pos(x, init_scales, threshold=1.5):
#     x = x.cpu()
#     scales = calc_scales(x).expand(-1, 3)
#     return torch.where(scales > threshold * init_scales, x + threshold * init_scales, x).cuda()


# init_scales = calc_scales(mesh.get_state().pos[mesh.particle_start:mesh.particle_end]).expand(-1, 3).cpu()

horizon = 150
for i in range(horizon + 1):
    scene.step()
    # if i % 2 == 0:
        # mesh.get_state().pos[mesh.particle_start:mesh.particle_end] = clamp_pos(mesh.get_state().pos[mesh.particle_start:mesh.particle_end], init_scales).cuda()
    torch.save(
        torch.tensor(mesh.get_state().pos[mesh.particle_start:mesh.particle_end].clone().detach().to("cpu")).reshape(-1, 3, 3),
        os.path.join(save_path, f"{i:05d}.pt")
    )

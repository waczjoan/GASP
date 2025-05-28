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

########################## create a scene ##########################

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=4e-3,
        substeps=10,
    ),
    mpm_options=gs.options.MPMOptions(
        # lower_bound=(-0.5, -1.0, 0.0),
        # upper_bound=(0.5, 1.0, 1),
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
os.makedirs(save_path, exist_ok=True)
SCALE=0.4
mesh = scene.add_entity(
    material=gs.materials.MPM.Elastic(sampler="random"),#(E=1e5, nu=0.1, rho=1000),
    morph=gs.morphs.Mesh(file=obj_path, convexify=False, decompose_nonconvex=False, scale=SCALE)
)
scene.build()
########################## build ##########################
horizon = 200
for i in range(horizon):
    scene.step()
    saved_tensor = torch.tensor(mesh.get_state().pos[mesh.particle_start:mesh.particle_end].clone().detach().to("cpu")).reshape(-1, 3, 3)
    torch.save(saved_tensor, os.path.join(save_path, f"{i:05d}.pt"))
    del saved_tensor
    torch.cuda.empty_cache()
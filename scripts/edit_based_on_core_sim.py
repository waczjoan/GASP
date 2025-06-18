#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys
sys.path.append("games_submodule")

import torch
import struct
import numpy as np
import os
import time
import json
from games_submodule.scene import Scene
from tqdm import tqdm
from os import makedirs
from games_submodule.renderer.gaussian_points_animated_renderer import render
import torchvision
from games_submodule.utils.general_utils import safe_state
from argparse import ArgumentParser
from games_submodule.arguments import ModelParams, PipelineParams, get_combined_args
from games_submodule.games import gaussianModelRender
import trimesh

def undotransform2origin(position_tensor, scale, original_mean_pos):
    return original_mean_pos + position_tensor / scale

def undoshift2center111(position_tensor):
    tensor111 = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    return position_tensor - tensor111

def particle_position_to_ply(mpm_solver, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = mpm_solver.mpm_state.particle_x.numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, "wb") as f:  # write binary
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)

def load_ply_to_tensor(filename):
    """
    Reads a PLY file and converts it back into a PyTorch tensor.
    Assumes the PLY file is in binary_little_endian format with float x, y, z properties.
    """
    with open(filename, 'rb') as f:
        # Read header
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == 'end_header':
                break

        # Extract the number of vertices
        num_vertices = 0
        for line in header:
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
                break

        if num_vertices == 0:
            raise ValueError("No vertex data found in the PLY file.")

        # Read binary vertex data
        vertex_data = f.read()
        position = np.frombuffer(vertex_data, dtype=np.float32)
        position = position.reshape((num_vertices, 3))

    # Convert to PyTorch tensor
    return torch.from_numpy(position)

def apply_inverse_rotation(position_tensor, rotation_matrix):
    rotated = torch.mm(position_tensor, rotation_matrix)
    return rotated

def apply_inverse_rotations(position_tensor, rotation_matrices):
    for i in range(len(rotation_matrices)):
        R = rotation_matrices[len(rotation_matrices) - 1 - i]
        position_tensor = apply_inverse_rotation(position_tensor, R)
    return position_tensor

def calculate_alpha(psuedomesh_triangles, core_triangles):
    v1 = core_triangles[:, 0,:]
    v2 = core_triangles[:, 1,:]
    v3 = core_triangles[:, 2,:]

    v2_v1 = v2 - v1
    v3_v1 = v3 - v1

    normal = torch.cross(v2_v1,v3_v1)
    v2_v1 = v2_v1 / torch.linalg.vector_norm(v2_v1, dim=-1, keepdim=True)
    v3_v1 = v3_v1 / torch.linalg.vector_norm(v3_v1, dim=-1, keepdim=True)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    A_T = torch.stack([normal, v2_v1, v3_v1]).permute(1, 2, 0)

    w1 = psuedomesh_triangles[:, 0,:]
    w2 = psuedomesh_triangles[:, 1,:]
    w3 = psuedomesh_triangles[:, 2,:]

    # calculate alpha
    alpha_w1 = torch.linalg.solve(A_T, w1 - v1).reshape(A_T.shape[0],3,1)
    alpha_w2 = torch.linalg.solve(A_T, w2 - v1).reshape(A_T.shape[0],3,1)
    alpha_w3 = torch.linalg.solve(A_T, w3 - v1).reshape(A_T.shape[0],3,1)
    alpha_w3.permute(0, 2, 1)

    return alpha_w1, alpha_w2, alpha_w3

def calc_new_vertices_position(alpha, normal, vec_1, vec_2, vertice_1):
    vertices = torch.bmm(
        alpha.permute(0, 2, 1),torch.stack((normal, vec_1, vec_2), dim=1)
    ).reshape(-1, 3)  + vertice_1
    return vertices
    

def calculate_sub_triangles(referenced_triangle, alpha_w1, alpha_w2, alpha_w3):
    v1_referenced = referenced_triangle[:, 0,:]
    v2_referenced = referenced_triangle[:, 1,:]
    v3_referenced = referenced_triangle[:, 2,:]

    referenced_v2_v1 = v2_referenced - v1_referenced
    referenced_v3_v1 = v3_referenced - v1_referenced
    normal = torch.cross(referenced_v2_v1, referenced_v3_v1)

    # norm
    referenced_v2_v1 = referenced_v2_v1 / torch.linalg.vector_norm(referenced_v2_v1, dim=-1, keepdim=True)
    referenced_v3_v1 = referenced_v3_v1 / torch.linalg.vector_norm(referenced_v3_v1, dim=-1, keepdim=True)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)


    # calculate new vertices of edited psuedomesh
    w1_edited = calc_new_vertices_position(alpha_w1, normal, referenced_v2_v1, referenced_v3_v1, v1_referenced)
    w2_edited = calc_new_vertices_position(alpha_w2, normal, referenced_v2_v1, referenced_v3_v1, v1_referenced)
    w3_edited = calc_new_vertices_position(alpha_w3, normal, referenced_v2_v1, referenced_v3_v1, v1_referenced)

    psuedomesh_edited_triangles = torch.stack(
        [w1_edited, w2_edited, w3_edited]
    ).permute(1,0,2)

    return psuedomesh_edited_triangles

def render_set_with_ply(gs_type, model_path, name, iteration, views, ply_files, gaussians, pipeline, background, scale_origin, original_mean_pos, camera_params, mask, unselected_pos, rotation_matrices, scale=1):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"our_core")
    core_path = os.path.join(model_path, "core_info", f"ours_{iteration}")

    psuedomesh = trimesh.load(os.path.join(model_path, "pseudomesh_info", f"ours_{iteration}", f"scale_{scale}.obj"), force='mesh')
    core_pseudomesh = trimesh.load(os.path.join(core_path, f"scale_{scale}.obj"), force='mesh')
    mapping = np.load(os.path.join(core_path, "pred.npy"))
    psuedomesh_triangles = torch.tensor(psuedomesh.triangles).cuda().float()

    core_pseudomesh_triangles = torch.tensor(core_pseudomesh.triangles).cuda().float()
    core_triangles = core_pseudomesh_triangles[mapping.flatten()]

    alphas = calculate_alpha(psuedomesh_triangles, core_triangles)

    makedirs(render_path, exist_ok=True)
    render_time = 0.0
    for idx, (ply_file) in enumerate(ply_files):
        view = views[0] #### TODO:CHANGE VIEW TO THE CAMERA IDX THAT YOU WANT TO USE
        print(f"Loading PLY file: {ply_file}")
        if mask is None:
            triangles = load_ply_to_tensor(ply_file).cuda()[:(psuedomesh_triangles.size(0)*3)]
            start_time = time.time()
            triangles = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(triangles), scale_origin, original_mean_pos
                ),
                rotation_matrices
            ).reshape(-1, 3, 3)
        else:
            triangles_selected = load_ply_to_tensor(ply_file).cuda()
            start_time = time.time()
            triangles_selected = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(triangles_selected), scale_origin, original_mean_pos
                ),
                rotation_matrices
            )
            triangles_unselected = unselected_pos.cuda()
            triangles = torch.empty((triangles_selected.shape[0] + triangles_unselected.shape[0], 3)).cuda()
            triangles[mask, :] = triangles_selected
            triangles[~mask, :] = triangles_unselected
            triangles = triangles.reshape(-1, 3, 3)

        core_triangles_edited = triangles[:, [0, 2, 1]] 

        referenced_triangle = core_triangles_edited[mapping.flatten()]

        sub_triangles = calculate_sub_triangles(referenced_triangle, *alphas)
        rendering = render(sub_triangles, view, gaussians, pipeline, background)["render"]
        render_time += time.time() - start_time
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

    with open(os.path.join(core_path, "time_info_our.json"), "r+") as file:
        data = json.load(file)
        data["fps"] = float(data["frame_num"]) / (float(data["sim_time"]) + render_time)
        data["render_time"] = render_time
        file.seek(0)
        json.dump(data, file)
        file.truncate()

    fps = int(1.0 / 4e-2)
    os.system(
        f"ffmpeg -framerate {fps} -i {render_path}/%05d.png -c:v libx264 -s {views[0].image_width // 2 * 2}x{views[0].image_height // 2 * 2} -y -pix_fmt yuv420p {render_path}/output.mp4"
    )

def render_sets_with_core_ply(gs_type: str, dataset: ModelParams, iteration: int, pipeline: PipelineParams, ply_dir: str, skip_train: bool, skip_test: bool, scale=1):
    with torch.no_grad():
        # Load all PLY files from the specified directory
        ply_files = [os.path.join(ply_dir, f) for f in sorted(os.listdir(ply_dir)) if f.endswith('.ply')]
        if not ply_files:
            raise ValueError(f"No PLY files found in directory: {ply_dir}")
        scale_origin_path = os.path.join(ply_dir, "scale_origin.pt")
        original_mean_pos_path = os.path.join(ply_dir, "original_mean_pos.pt")
        rotation_matrices = torch.load(os.path.join(ply_dir, "rotation_matrices.pt"))
        mask_path = os.path.join(ply_dir, "mask.pt")
        if os.path.exists(mask_path):
            mask = torch.load(mask_path)
            unselected_pts = torch.load(os.path.join(ply_dir, "unselected_pos.pt"))
        else:
            mask = None
            unselected_pts = None
        scale_origin = torch.load(scale_origin_path)
        original_mean_pos = torch.load(original_mean_pos_path)
        with open(os.path.join(ply_dir, "camera_params.json"), "r") as file:
            import json
            camera_params = json.load(file)

        gaussians = gaussianModelRender[gs_type](dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set_with_ply(gs_type, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), ply_files, gaussians, pipeline, background, scale_origin, original_mean_pos, camera_params, mask, unselected_pts, rotation_matrices, scale)

        if not skip_test:
            render_set_with_ply(gs_type, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), ply_files, gaussians, pipeline, background, scale_origin, original_mean_pos, camera_params, mask, unselected_pts, rotation_matrices, scale)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    # parser.add_argument("--ply_dir", type=str, required=True, help="Directory containing PLY files to render.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    args.ply_dir = os.path.join(args.model_path, "core_info", "ours_30000", "simulation_ply") 
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets_with_core_ply(args.gs_type, model.extract(args), args.iteration, pipeline.extract(args), args.ply_dir, args.skip_train, args.skip_test, args.scale)

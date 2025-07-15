#
# Copyright (C) 2025, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import sys
sys.path.append("games_submodule")

import torch
from games_submodule.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from games_submodule.renderer.gaussian_points_animated_renderer import render
import torchvision
import trimesh
from games_submodule.utils.general_utils import safe_state
from argparse import ArgumentParser
from games_submodule.arguments import ModelParams, PipelineParams, get_combined_args
from games_submodule.games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel


def transform_vertices_function(vertices, c=1):
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def transform_diff(vertices, vertices_diff, t):
    vertices += vertices_diff * t
    return vertices


def do_not_transform(vertices, t):
    return vertices


def render_set(
        model_path, name, iteration, views, gaussians,
        pipeline, background, sim_dirname, skip_sym_obj, scale, camera_view=4
    ):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), os.path.basename(sim_dirname))
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    pseudomesh_info_path = os.path.join(model_path, "pseudomesh_info", "ours_{}".format(iteration))
    #makedirs(f"{sim_dirname}/objects", exist_ok=True)
    #makedirs(f"{sim_dirname}/objects/scale_{scale}", exist_ok=True)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    faces = torch.load(f"{pseudomesh_info_path}/faces.pt")
    view = views[camera_view]

    lst = os.listdir(f"{sim_dirname}")  # your directory path
    number_files = len(lst)
    for idx in range(0, number_files):
        _idx = '{0:04d}'.format(idx * skip_sym_obj)
        try:
            mesh_scene = trimesh.load(f'{sim_dirname}/{_idx}.obj', force='mesh')
            vertice = transform_vertices_function(torch.tensor(mesh_scene.vertices)).float()
        except:
            mesh_scene = torch.load(f'{sim_dirname}/{_idx}.pt')
            vertice = transform_vertices_function(torch.tensor(mesh_scene)).float().cuda()
        triangles = vertice[faces.long()].cuda()
        rendering = render(triangles, view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, _idx + ".png"))



def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)

def render_sets(
        dataset : ModelParams, iteration : int, pipeline : PipelineParams,
        skip_train : bool, skip_test : bool, sim_dirname, skip_sym_obj, scale
    ):
    with torch.no_grad():
        gaussians = PointsGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(
                 dataset.model_path, "train",
                 scene.loaded_iter, scene.getTrainCameras(),
                 gaussians, pipeline, background, sim_dirname,
                 skip_sym_obj, scale
             )

        if not skip_test:
             render_set(
                 dataset.model_path, "test",
                 scene.loaded_iter, scene.getTestCameras(),
                 gaussians, pipeline, background, sim_dirname,
                skip_sym_obj, scale
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    parser.add_argument('--sim_dirname', type=str)
    parser.add_argument('--skip_sym_obj', type=int, default=1)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)


    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args.sim_dirname,
        args.skip_sym_obj,
        args.scale
    )

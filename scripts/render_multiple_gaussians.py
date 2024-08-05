#
# Copyright (C) 2024, Gmum
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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from renderer import render
import torchvision
import trimesh
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
import copy


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
        model_paths, name, iteration, view, gaussians_list,
        pipeline, background, sym_dirnames, skip_sym_obj, scale
    ):
    render_path = os.path.join(
        model_paths[0], name, "ours_multiple", os.path.basename(sym_dirnames[0])
    )
    #gts_path = os.path.join(model_paths[0], name, "ours_{}".format(iteration), "gt")

    #gaussians = copy.deepcopy(gaussians_list[0])

    for name in sym_dirnames:
        makedirs(f"{name}/objects", exist_ok=True)
        makedirs(f"{name}/objects/scale_{scale}", exist_ok=True)

    makedirs(render_path, exist_ok=True)
    #makedirs(gts_path, exist_ok=True)

    faces = {}
    for model_path in model_paths:
        pseudomesh_info_path = os.path.join(model_path, "pseudomesh_info", "ours_{}".format(iteration[0]))
        _faces = torch.load(f"{pseudomesh_info_path}/faces.pt")
        faces[model_path] = _faces

    lst = os.listdir(f"{sym_dirnames[0]}/triangles")  # your directory path
    number_files = len(lst)
    for idx in range(0, number_files, skip_sym_obj):
        _idx = '{0:04d}'.format(idx)

        triangles = []
        for i, sym_dirname in zip(model_paths, sym_dirnames):
            try:
                mesh_scene = trimesh.load(f'{sym_dirname}/objects/scale_{scale}/{_idx}.obj', force='mesh')
                _triangles = mesh_scene.triangles/scale
            except:
                vertice = torch.tensor(torch.load(f'{sym_dirname}/triangles/{_idx}.pt'))
                vertice = transform_vertices_function(vertice)
                filename = f'{sym_dirname}/objects/scale_{scale}/{_idx}.obj'
                write_simple_obj(mesh_v=(vertice * scale).detach().cpu().numpy(), mesh_f=faces[i], filepath=filename)
                _triangles = vertice[faces[i].long()].cuda()
            _traingles = torch.tensor(_triangles).float().cuda()
            triangles.append(_traingles)

        #traingles = torch.hstack(triangles)
        rendering = render(triangles, view, gaussians_list, pipeline, background)["render"]
        #gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        #torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)

def render_sets(
        model_paths, datasets : ModelParams, iteration : int, pipeline : PipelineParams,
        skip_train : bool, skip_test : bool, sym_dirname, skip_sym_obj, scale
    ):
    with torch.no_grad():
        gaussians_list = []
        scene_list = []
        loaded_iters = []
        for dataset in datasets:
            gaussians = PointsGaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            if hasattr(gaussians, 'prepare_vertices'):
                gaussians.prepare_vertices()
            if hasattr(gaussians, 'prepare_scaling_rot'):
                gaussians.prepare_scaling_rot()
            gaussians_list.append(copy.deepcopy(gaussians))
            if dataset == datasets[0]:
                train_view = scene.getTrainCameras()[4]
                test_view = scene.getTestCameras()[4]
            loaded_iters.append(scene.loaded_iter)
            del scene
            #scene_list.append(scene)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(
                 model_paths, "train",
                 loaded_iters, train_view,
                 gaussians_list, pipeline, background, sym_dirname,
                 skip_sym_obj, scale
             )

        if not skip_test:
             render_set(
                 model_paths, "test",
                 loaded_iters, test_view,
                 gaussians_list, pipeline, background, sym_dirname,
                 skip_sym_obj, scale
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    parser.add_argument('--sym_dirnames', nargs="+", type=str, default=[])
    parser.add_argument('--skip_sym_obj', type=int, default=8)
    parser.add_argument('--scale', type=int, default=100)
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--model_paths", nargs="+", type=str, default=[])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    args = []
    sym_dirname = {}
    for i, sym_dirname_i in zip(arguments.model_paths, arguments.sym_dirnames):
        parser.__setattr__("model_path", i)
        args.append(get_combined_args(parser))
        sym_dirname[i] = sym_dirname_i
    model.gs_type = args[0].gs_type
    print("Rendering " + str(arguments.model_paths))

    # Initialize system state (RNG)
    safe_state(args[0].quiet)

    datasets = [model.extract(arg) for arg in args]


    render_sets(
        arguments.model_paths,
        datasets,
        arguments.iteration,
        pipeline.extract(args[0]),
        arguments.skip_train,
        arguments.skip_test,
        arguments.sym_dirnames,
        arguments.skip_sym_obj,
        arguments.scale
    )

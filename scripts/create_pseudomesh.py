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

import torch
from loader import GaussiansLoader
import os
from os import makedirs
from games.utils.general_utils import safe_state
from argparse import ArgumentParser
from games.arguments import ModelParams, PipelineParams, get_combined_args
from games.games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel


def render_sets(modelparams : ModelParams, iteration : int):
    with torch.no_grad():
        gaussians = PointsGaussianModel(modelparams.sh_degree)
        model = GaussiansLoader(modelparams, gaussians, load_iteration=iteration)
        print("dipa")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_points")

    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Pseudomesh info " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration)
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
from loader import GaussiansLoader
from games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
from argparse import ArgumentParser


def save_pseudomesh_info(sh_degree, model_path, iteration : int):
    with torch.no_grad():
        gaussians = PointsGaussianModel(sh_degree)
        model = GaussiansLoader(model_path, gaussians, load_iteration=iteration)
        print("dipa")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_points")
    args = parser.parse_args()

    print("Pseudomesh info " + args.model_path)

    model_path = args.model_path

    save_pseudomesh_info(args.sh_degree, args.model_path, args.iteration)
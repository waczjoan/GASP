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
# This Gape software is free for non-commercial, research and evaluation use
#

import os
from games.utils.system_utils import searchForMaxIteration
from games.scene.gaussian_model import GaussianModel
from games.arguments import ModelParams

class GaussiansLoader:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration):
        """b
        :param path: Path to colmap loader main folder.
        """
        self.model_path = args.model_path
        self.gaussians = gaussians

        if load_iteration == -1:
            self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        else:
            self.loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.gaussians.load_ply(
            os.path.join(self.model_path,
            "point_cloud",
            "iteration_" + str(self.loaded_iter),
            "point_cloud.ply")
        )


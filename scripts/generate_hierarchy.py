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

import os
import torch
from os import makedirs
from games_submodule.games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
from games_submodule.games.flat_splatting.scene.flat_gaussian_model import FlatGaussianModel
from games_submodule.utils.system_utils import searchForMaxIteration
from games_submodule.scene.gaussian_model import GaussianModel
from argparse import ArgumentParser
from games_submodule.utils.graphics_utils import BasicPointCloud
from games_submodule.utils.sh_utils import SH2RGB

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import Birch
import torch
#from sklearn.preprocessing import normalize

class GaussiansLoader:

    gaussians : GaussianModel

    def __init__(self, model_path, gaussians : GaussianModel, load_iteration):
        """b
        :param path: Path to colmap loader main folder.
        """
        self.model_path = model_path
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

        if hasattr(self.gaussians, 'prepare_vertices'):
            self.gaussians.prepare_vertices()
        if hasattr(self.gaussians, 'prepare_scaling_rot'):
            self.gaussians.prepare_scaling_rot()


def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)

def calculate_cov(cluster):
    '''
    Calculate covariance matrix using cluster points. 

    Parameters:
        cluster (np.ndarray, (N, 3)): the xyz points from one cluster 

    Returns:
        np.ndarray (3, 3): the covariance matrix for specific cluster
    '''
    m = cluster.mean(axis=0)
    x = cluster - m
    return 1/(cluster.shape[0] - 1) * x.T @ x

def calculate_core_gaussians(xyz, rots, scals, clustering_alg, eps=1e-8):
    ''' 
    Calculate the mean and the covariance for core gaussians from clustering.

    Parameters:
        xyz (np.ndarray, (N, 3)): means from 3DGS model
        rots (np.ndarray, (N, 4)): quaternion from 3DGS model
        scals (np.ndarray, (N, 3)): scaling from 3DGS model
        clustering_alg (sklearn.cluster model): fitted clustering model from sklearn

    Returns:
        new_xyz (np.ndarray, (M, 3)): new mean for subgaussians
        new_rots (np.ndarray, (M, 4)): new quaternion for subgaussians
        new_scals (np.ndarray, (M, 3)): new scaling for subgaussians
        mapping (np.ndarray, (N, 1)): position of the core gaussian for every sub gaussian
    '''
    pred = clustering_alg.predict(xyz)
    new_m = clustering_alg.subcluster_centers_
    new_xyz = []
    new_rots = []
    new_scals = []
    mapping = np.zeros(pred.shape)
    n = 0

    for i in range(len(np.unique(pred))):
        p = xyz[pred==i]
        if len(p) == 1:
            new_rots.append(rots[pred==i][0])
            new_scals.append(scals[pred==i][0])
            new_xyz.append(p[0])
            mapping[pred==i] = i - n
            continue
        elif len(p) == 0:
            n += 1
            continue

        # Change from RS to GaMeS
        triangles = change_from_RS_to_games(p, rots[pred==i], scals[pred==i])
        
        # Take points from GaMeS and calculate covariance matrix for cluster
        points = triangles.reshape((-1, 3)).numpy()
        cov = calculate_cov(points)

        # Change from covariance matrix to RS
        rot, scal = change_from_cov_to_RS(cov)
        scal[0] = eps
        new_rots.append(rot)
        new_scals.append(scal)
        new_xyz.append(new_m[i])
        mapping[pred==i] = i - n


    return np.stack(new_xyz), np.stack(new_rots), np.stack(new_scals), mapping


def change_from_cov_to_RS(cov):
    ''' 
    Change Gaussian with covariance matrix into quaternion and scaling.

    Parameters:
        cov (np.ndarray, (N, 3, 3)): covariance matrices 

    Returns:
        rots (np.ndarray, (N, 4)): quaternions
        scals (np.ndarray, (N, 3)): scaling
    '''
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    rots = eigenvectors
    rots = R.from_matrix(rots).as_quat()
    scalar_first = np.roll(rots, shift=-1, axis=-1)
    scals = np.sqrt(eigenvalues)
    return scalar_first, scals
    

def change_from_RS_to_games(means, rots, scals):
    ''' 
    Change from rotation and scaling vectors to GaMeS representation
    
    Parameters:
        means (np.ndarray, (N, 3)): Gaussian means
        rots (np.ndarray, (N, 4)): quaternion 
        scals (np.ndarray, (N, 3)): scaling
        
    Returns:
        triangles (torch.tensor, (N, 3, 3)): GaMeS triangle soup, (N, points_in_triangle, xyz)
    '''
    scalar_first = np.roll(rots, shift=-1, axis=-1)
    rots_mat = R.from_quat(scalar_first).as_matrix()
    scals = np.expand_dims(scals, axis=2)
    VS = np.array([means, means + scals[:,1]*rots_mat[:,:,1], means + scals[:,2]*rots_mat[:,:,2]])
    VS = torch.tensor(VS)
    return VS.permute((1,0,2))


def generate_hierarchy(
        sh_degree,
        model_path,
        iteration : int,
        scale = 1,
        threshold = 0.1
):
    with torch.no_grad():
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        gaussians = PointsGaussianModel(sh_degree)
        model = GaussiansLoader(model_path, gaussians, load_iteration=iteration)
        core_info_path = os.path.join(model_path, "core_info", "ours_{}".format(model.loaded_iter))
        makedirs(core_info_path, exist_ok=True)

        birch = Birch(threshold=threshold, n_clusters=None)
        print("Fitting Birch clustering")
        points = model.gaussians.get_xyz.detach().cpu().numpy()
        rots = model.gaussians.get_rotation.detach().cpu().numpy()
        scals = model.gaussians.get_scaling.detach().cpu().numpy()
        birch.fit(points)

        new_xyz, new_rot, new_scal, mapping = calculate_core_gaussians(points, rots, scals, birch, gaussians.eps_s0)

        triangles = change_from_RS_to_games(new_xyz, new_rot, new_scal)
        print("Saving mapping between core and sub Gaussians")

        print(f"Triangles shape: {triangles.shape}")
        faces = torch.range(0, triangles.shape[0] * 3 - 1).reshape(triangles.shape[0], 3)
        vertices = triangles.reshape(triangles.shape[0] * 3, 3)
        torch.save(triangles, os.path.join(core_info_path, 'triangles.pt'))
        torch.save(faces, os.path.join(core_info_path, 'faces.pt'))
        torch.save(vertices, os.path.join(core_info_path, 'vertices.pt'))
        SCALE = scale
        filename = os.path.join(core_info_path, f'scale_{SCALE}.obj')
        write_simple_obj(mesh_v=(vertices * SCALE).detach().cpu().numpy(), mesh_f=faces, filepath=filename)

        with open(os.path.join(core_info_path, "pred.npy"), "wb") as f:
            np.save(f, mapping)


        new_scal = torch.from_numpy(new_scal)
        new_rot = torch.from_numpy(new_rot)
        xyz = new_xyz
        shs = np.random.random((xyz.shape[0], 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))
        core_gauss = FlatGaussianModel(sh_degree)
        core_gauss.create_from_pcd(pcd, gaussians.spatial_lr_scale)
        core_gauss._scaling = torch.log(new_scal)
        core_gauss._rotation = new_rot

        core_point_cloud_path = os.path.join(core_info_path, "point_cloud/iteration_{}".format(iteration))
        core_gauss.save_ply(os.path.join(core_point_cloud_path, "point_cloud.ply"))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--scale", default=1, type=int)
    parser.add_argument("--threshold", default=0.1, type=int)

    args = parser.parse_args()


    print("Pseudomesh info " + args.model_path)

    model_path = args.model_path

    generate_hierarchy(
        args.sh_degree,
        args.model_path,
        args.iteration,
        args.scale,
        args.threshold
    )
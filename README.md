# GASP: Gaussian Splatting for Physics-Based Simulations

Physical simulation is paramount for modeling and
utilization of 3D scenes in various real-world
applications. However, its integration with state-of-the-art 3D
scene rendering techniques such as Gaussian Splatting (GS) remains
challenging. Existing models use additional meshing mechanisms,
including triangle or tetrahedron meshing, marching cubes, or cage meshes.
As an alternative, we can modify the physically grounded Newtonian dynamics
to align with 3D Gaussian components. Current models take the first-order
approximation of a deformation map, which locally approximates the dynamics by
linear transformations. In contrast, our Gaussian Splatting for Physics-Based
Simulations (GASP) model uses such a map (without any modifications) and
flat Gaussian distributions, which are parameterized by three points (mesh faces).
Subsequently, each 3D point (mesh face node) is treated as a discrete entity within a 3D space.
Consequently, the problem of modeling Gaussian components is reduced to working with 3D points.
Additionally, the information on mesh faces can be used to incorporate further properties into
the physical model, facilitating the use of triangles. Resulting solution can be integrated into
any physical engine that can be treated as a black box. As demonstrated in our studies,
the proposed model exhibits superior performance on a diverse
range of benchmark datasets designed for 3D object rendering.

Project Page under the [LINK](https://waczjoan.github.io/GASP/).
Arxiv [LINK](https://arxiv.org/abs/2409.05819)

</br>
<img src="./docs/static/videos/car/car.gif" width="250" height="250"/>
<img src="./docs/static/videos/more/cup.gif" width="250" height="250"/>
</br>


Wait for it...

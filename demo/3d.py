import taichi as ti
import numpy as np
import torch
import utils
from engine.mpm_solver import MPMSolver
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', type=str, help='Input folder')
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    parser.add_argument('--material', type=str, help='material type')
    parser.add_argument('--E', type=float, help='youngs modulus')
    parser.add_argument('--threshold', type=float, default=1.0, help='threshold')
    parser.add_argument('--skip', type=int, default=8)
    args = parser.parse_args()
    print(args)
    return args

args = parse_args()

threshold = args.threshold

def save_positions_pt(positions, iteration):
    positions = scaler.inverse(positions)
    positions_tensor = torch.from_numpy(positions)
    filename = args.out_dir + f'/triangles/{iteration:04d}.pt'
    torch.save(positions_tensor, filename)

class Rescale:
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
    
    def transform(self, x):
        return 0.5 * (x - self.min) / (self.max - self.min) + np.array([0.25, 0.5, 0.25])
    
    def inverse(self, x):
        return 2 * x * (self.max - self.min)

ti.init(arch=ti.gpu, device_memory_fraction=0.9) 

material_list = {
    'elastic': MPMSolver.material_elastic,
    'sand': MPMSolver.material_sand,
    'snow': MPMSolver.material_snow,
    'water': MPMSolver.material_water
}

material = material_list[args.material]
write_to_disk = args.out_dir is not None
if write_to_disk:
    os.makedirs(f'{args.out_dir}/triangles', exist_ok=True)
    os.makedirs(f'{args.out_dir}/img', exist_ok=True)

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41, show_gui=False)

pts = torch.load(f'{args.in_dir}/vertices.pt').cpu().numpy()
pts[:, 1] = -pts[:, 1]
pts = pts[:, [0, 2, 1]]
scaler = Rescale()
scaler.fit(pts)
pts = scaler.transform(pts)

mpm = MPMSolver(res=(64, 64, 64), E_scale=args.E)

mpm.add_particles(particles=pts,
                material=material)

init_scales = ti.field(dtype=ti.f32, shape=mpm.n_particles[None])
new_scales = ti.field(dtype=ti.f32, shape=mpm.n_particles[None])

def calc_scales(x):
    scales = x.reshape((-1, 3, 3))
    scales = scales - np.expand_dims(scales[:, 0, :], -2)
    scales = np.linalg.norm(scales, axis=-1)
    return scales.flatten()

@ti.kernel
def modify_positions():
    for i in range(mpm.n_particles[None] // 3):
        m = mpm.x[3 * i]
        for j in ti.static(range(1, 3)):
            idx = 3 * i + j
            diff = mpm.x[idx] - m
            norm = diff.norm()
            v = diff / norm
            if new_scales[idx] / init_scales[idx] > threshold:
                mpm.x[idx] = m + threshold * init_scales[idx] * v

for frame in range(200):
    mpm.step(1e-2)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                    dtype=np.uint32)
    particles = mpm.particle_info()
    new_scales.from_numpy(calc_scales(particles['position']))
    modify_positions()
    particles = mpm.particle_info()
    np_x = particles['position']
    screen_x = (np_x[:, 0]) #((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])
    screen_pos = np.stack([screen_x, screen_y], axis=-1)
    if frame % args.skip == 0:
        save_positions_pt(particles['position'], frame)
    gui.circles(screen_pos,
                radius=1.5,
                color=colors[particles['material']])
    gui.show(f'{args.out_dir}/img/{frame:06d}.png' if write_to_disk else None)

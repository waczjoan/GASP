import taichi as ti
import numpy as np
import torch
import utils
from engine.mpm_solver import MPMSolver
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-dir', type=str, nargs='+', default=[], help='Input folder')
    parser.add_argument('-o', '--out-dir', type=str, nargs='+', default=[], help='Output folder')
    parser.add_argument('--scales', type=float, nargs='+')
    parser.add_argument('--offsets', type=float, nargs='+')
    parser.add_argument('--material', type=str, default='elastic', help='material type')
    parser.add_argument('--E', type=float, default=1.0, help='youngs modulus')
    parser.add_argument('--threshold', type=float, default=1.0, help='threshold')
    parser.add_argument('--skip', type=int, default=8)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    print(args)
    return args

args = parse_args()

scales = args.scales if len(args.scales) == 2 else [args.scales[:3], args.scales[3:]]
offsets = [args.offsets[:3], args.offsets[3:]]
threshold = args.threshold

def save_positions_pt(positions, iteration):
    positions = np.split(positions, [len(pts_list[0])])
    for i, x in zip(range(2), positions):
        x = scaler_list[i].inverse(x, scaler_list[1].scale, scaler_list[1].max,  scaler_list[1].min, scaler_list[1].offset)
        positions_tensor = torch.from_numpy(x)
        filename = args.out_dir[i] + f'/triangles/{iteration:04d}.pt'
        torch.save(positions_tensor, filename)

class Rescale:
    def __init__(self, scale=0.5, offset=[0.25, 0.5, 0.25]):
        self.min = None
        self.max = None
        self.scale = scale
        self.offset = np.array(offset)
    
    def fit(self, x):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
    
    def transform(self, x):
        return self.scale * (x - self.min) / (self.max - self.min) + self.offset
    
    def inverse(self, x, new_scale=None, new_max=None, new_min=None, new_offset=None):
        scale = self.scale if new_scale is None else new_scale
        new_max = self.max if new_max is None else new_max
        new_min = self.min if new_min is None else new_min
        offset = self.offset if new_offset is None else new_offset
        return (x - offset) / scale * (new_max - new_min) + new_min

def get_points(scales, offsets):
    pts_list = []
    scaler_list = []
    for path, scale, offset in zip(args.in_dir, scales, offsets):
        pts = torch.load(f'{path}/vertices.pt').cpu().numpy()
        pts[:, 1] = -pts[:, 1]
        pts = pts[:, [0, 2, 1]]
        scaler = Rescale(scale, offset)
        scaler.fit(pts)
        pts = scaler.transform(pts)
        pts_list.append(pts)
        scaler_list.append(scaler)
    return pts_list, scaler_list

write_to_disk = args.out_dir[0] is not None
if write_to_disk:
    for out_dir in args.out_dir:
        os.makedirs(f'{out_dir}/triangles', exist_ok=True)
        os.makedirs(f'{out_dir}/img', exist_ok=True)

ti.init(arch=ti.gpu, device_memory_fraction=0.9) 

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41, show_gui=False)

mpm = MPMSolver(res=(64, 64, 64), E_scale=args.E)

pts_list, scaler_list = get_points(
    scales=scales,
    offsets=offsets
)

material_list = {
    'elastic': MPMSolver.material_elastic,
    'sand': MPMSolver.material_sand,
    'snow': MPMSolver.material_snow,
    'water': MPMSolver.material_water
}

material = material_list[args.material]
mpm.add_particles(particles=pts_list[0],
                material=material)

mpm.add_particles(particles=pts_list[1],
                material=MPMSolver.material_stationary)

init_scales = ti.field(dtype=ti.f32, shape=mpm.n_particles[None])
new_scales = ti.field(dtype=ti.f32, shape=mpm.n_particles[None])

def calc_scales(pts):
    if isinstance(pts, list):
        pts = np.concatenate(pts)
    x = pts
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
            elif init_scales[idx] / new_scales[idx] > threshold:
                mpm.x[idx] = m + 1.0 / threshold * init_scales[idx] * v

init_scales.from_numpy(calc_scales(pts_list))

for frame in range(args.iters):
    mpm.step(1e-2)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00, 0xCD553B],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    new_scales.from_numpy(calc_scales(particles['position']))
    modify_positions()
    particles = mpm.particle_info()
    np_x = particles['position']
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])
    screen_pos = np.stack([screen_x, screen_y], axis=-1)
    if frame % args.skip == 0:
        save_positions_pt(particles['position'], frame)
    gui.circles(screen_pos,
                radius=1.5,
                color=colors[particles['material']])
    for out_dir in args.out_dir:
        gui.show(f'{out_dir}/img/{frame:06d}.png' if write_to_disk else None)

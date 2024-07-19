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
    args = parser.parse_args()
    print(args)
    return args

args = parse_args()

def save_positions_pt(positions, iteration):
    positions = scaler.inverse(positions)
    positions_tensor = torch.from_numpy(positions)
    filename = args.out_dir + f'/positions/{iteration:04d}.pt'
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
        return 2 * (x - np.array([0.25, 0.5, 0.25])) * (self.max - self.min) + self.min

write_to_disk = args.out_dir is not None
if write_to_disk:
    os.makedirs(f'{args.out_dir}/positions', exist_ok=True)
    os.makedirs(f'{args.out_dir}/img', exist_ok=True)

ti.init(arch=ti.gpu, device_memory_GB=12) 
# ti.init()

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

pts = torch.load(f'{args.in_dir}/gs_flat____vertices_fish_cup.pt').cpu().numpy()
pts[:, 1] = -pts[:, 1]
pts = pts[:, [0, 2, 1]]
scaler = Rescale()
scaler.fit(pts)
pts = scaler.transform(pts)

mpm = MPMSolver(res=(128, 128, 128), E_scale=7e3)

mpm.add_particles(particles=pts,
                  material=MPMSolver.material_elastic)

for frame in range(200):
    mpm.step(1e-2)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    np_x = particles['position']
    screen_x = (np_x[:, 0]) #((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])
    screen_pos = np.stack([screen_x, screen_y], axis=-1)
    save_positions_pt(particles['position'], frame)
    gui.circles(screen_pos,
                radius=1.5,
                color=colors[particles['material']])
    gui.show(f'{args.out_dir}/img/{frame:06d}.png' if write_to_disk else None)

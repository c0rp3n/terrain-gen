import numpy as np
import random

from noise import snoise2
from utils import (create_dir, save_heightfield)

DIR = 'heightmaps/'

SCALE = 0.00125
WSCALE = 4.0
OCTAVES = 8

def noise_w0(x, y) -> float:
    # also times by WSCALE to match scale of terrain features
    return snoise2(x * SCALE * WSCALE, y * SCALE * WSCALE, OCTAVES)

def noise_w1(x, y) -> float:
    p_x = x * SCALE
    p_y = y * SCALE

    q_x = snoise2(p_x, p_y, OCTAVES)
    q_y = snoise2(p_x + 5.2, p_y + 1.3, OCTAVES)

    return snoise2(p_x + (WSCALE * q_x), p_y + (WSCALE * q_y), OCTAVES)

def noise_w2(x, y) -> float:
    # divide x by WSCALE to try match terrain features
    p_x = (x * SCALE) / WSCALE
    p_y = (y * SCALE) / WSCALE

    q_x = snoise2(p_x, p_y, OCTAVES)
    q_y = snoise2(p_x + 5.2, p_y + 1.3, OCTAVES)

    r_x = snoise2(p_x + (WSCALE * q_x) + 1.7, p_y + (WSCALE * q_y) + 9.2, OCTAVES)
    r_y = snoise2(p_x + (WSCALE * q_x) + 8.3, p_y + (WSCALE * q_y) + 2.8, OCTAVES)

    return snoise2(p_x + (WSCALE * r_x), p_y + (WSCALE * r_y), OCTAVES)

def main():
    create_dir(DIR)

    hf = np.zeros((512, 512), dtype=np.float32)

    for x in range(0, 512):
        for y in range(0, 512):
            hf[x, y] = noise_w0(x, y)

    save_heightfield(hf, DIR + 'hf_domain_warping_w0.png')

    for x in range(0, 512):
        for y in range(0, 512):
            hf[x, y] = noise_w1(x, y)

    save_heightfield(hf, DIR + 'hf_domain_warping_w1.png')

    for x in range(0, 512):
        for y in range(0, 512):
            hf[x, y] = noise_w2(x, y)

    save_heightfield(hf, DIR + 'hf_domain_warping_w2.png')

    return

if __name__ == "__main__":
    main()

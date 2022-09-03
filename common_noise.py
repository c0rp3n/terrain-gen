import numpy as np
import random

from noise import (snoise2, pnoise2)
from utils import (create_dir, save_heightfield)

DIR = 'heightmaps/'

OFFSET_X = 52.25
OFFSET_Y = 13.28
SCALE = 0.0125
OCTAVES = 1

def snoise(x, y) -> float:
    return snoise2((x * SCALE) + OFFSET_X, (y * SCALE) + OFFSET_Y, OCTAVES)

def pnoise(x, y) -> float:
    return pnoise2((x * 1.8 * SCALE) + OFFSET_X, (y * 1.8 * SCALE) + OFFSET_Y, OCTAVES)

def main():
    create_dir(DIR)

    hf = np.zeros((512, 512), dtype=np.float32)

    for x in range(0, 512):
        for y in range(0, 512):
            hf[x, y] = snoise(x, y)

    save_heightfield(hf, DIR + 'hf_simplex.png')

    for x in range(0, 512):
        for y in range(0, 512):
            hf[x, y] = pnoise(x, y)

    save_heightfield(hf, DIR + 'hf_perlin.png')

    return

if __name__ == "__main__":
    main()

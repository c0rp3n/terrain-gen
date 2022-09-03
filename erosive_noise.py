import numpy as np
import random

from sdnoise import sdnoise2
from utils import (create_dir, save_heightfield)

DIR = 'heightmaps/'

SIZE = 2048
OFFSET_X = 53.36
OFFSET_Y = 138.64

SCALE = 0.0025
PERSISTANCE = 0.5
LUNACRACITY = 2.0
OCTAVES = 6

# control the erosive noise
ALPHA = 0.15 # feature displacement
BETA = 1.1 # roughness near valeys

def fbm_noise(x, y) -> float:
    px = x * SCALE
    py = y * SCALE

    freq = 1.0
    amp = 1.0
    B = 0.0

    for j in range(OCTAVES):
        pre_x = freq * px
        pre_y = freq * py

        n, _, _ = sdnoise2(pre_x, pre_y)
        B = B + (amp * n)

        amp = amp * PERSISTANCE
        freq = freq * LUNACRACITY

    return B

def billowy_noise(x, y) -> float:
    px = x * SCALE
    py = y * SCALE

    freq = 1.0
    amp = 1.0
    B = 0.0

    for j in range(OCTAVES):
        pre_x = freq * px
        pre_y = freq * py

        n, _, _ = sdnoise2(pre_x, pre_y)
        B = B + (amp * abs(n))

        amp = amp * PERSISTANCE
        freq = freq * LUNACRACITY

    return B

def ridged_noise(x, y) -> float:
    px = x * SCALE
    py = y * SCALE

    freq = 1.0
    amp = 1.0
    B = 0.0

    for j in range(OCTAVES):
        pre_x = freq * px
        pre_y = freq * py

        n, _, _ = sdnoise2(pre_x, pre_y)
        B = B + (amp * (1.0 - abs(n)))

        amp = amp * PERSISTANCE
        freq = freq * LUNACRACITY

    return B

def erosive_noise(x, y) -> float:
    px = x * SCALE
    py = y * SCALE

    freq = 1.0
    amp = 1.0
    B = 0.0
    dx = 0.0
    dy = 0.0
    s = 1.0

    for j in range(OCTAVES):
        pre_x = freq * (px + dx)
        pre_y = freq * (py + dy)

        n, gx, gy = sdnoise2(pre_x, pre_y)
        t_in = s * (1 - abs(n))
        B = B + (amp * t_in)

        # aproximate gradient because 1 - abs(N(p))
        grad_x = -n * gx
        grad_y = -n * gy

        dx = dx + (amp * ALPHA * s * grad_x)
        dy = dy + (amp * ALPHA * s * grad_y)
        s = s * min(1, max(0, BETA * B))

        amp = amp * PERSISTANCE
        freq = freq * LUNACRACITY

    return B

def erosive_billowed_noise(x, y) -> float:
    px = x * SCALE
    py = y * SCALE

    freq = 1.0
    amp = 1.0
    B = 0.0
    dx = 0.0
    dy = 0.0
    s = 1.0

    for j in range(OCTAVES):
        pre_x = freq * (px + dx)
        pre_y = freq * (py + dy)

        n, gx, gy = sdnoise2(pre_x, pre_y)
        t_in = s * abs(n)
        B = B + (amp * t_in)

        dx = dx + (amp * ALPHA * s * gx)
        dy = dy + (amp * ALPHA * s * gy)
        s = s * min(1, max(0, BETA * B))

        amp = amp * PERSISTANCE
        freq = freq * LUNACRACITY

    return B

def main():
    create_dir(DIR)

    hf = np.zeros((SIZE, SIZE), dtype=np.float32)
#
    #for x in range(0, SIZE):
    #    for y in range(0, SIZE):
    #        hf[x, y] = fbm_noise(x + OFFSET_X, y + OFFSET_Y)
#
    #save_heightfield(hf, DIR + 'hf_fbm_noise.png')
#
    #for x in range(0, SIZE):
    #    for y in range(0, SIZE):
    #        hf[x, y] = billowy_noise(x + OFFSET_X, y + OFFSET_Y)
#
    #save_heightfield(hf, DIR + 'hf_billoy_noise.png')
#
    #for x in range(0, SIZE):
    #    for y in range(0, SIZE):
    #        hf[x, y] = ridged_noise(x + OFFSET_X, y + OFFSET_Y)
#
    #save_heightfield(hf, DIR + 'hf_ridged_noise.png')

    for x in range(0, SIZE):
        for y in range(0, SIZE):
            hf[x, y] = erosive_noise(x + OFFSET_X, y + OFFSET_Y)

    save_heightfield(hf, DIR + 'hf_erosive_noise.png')

    #for x in range(0, SIZE):
    #    for y in range(0, SIZE):
    #        hf[x, y] = erosive_billowed_noise(x + OFFSET_X, y + OFFSET_Y)
#
    #save_heightfield(hf, DIR + 'hf_erosive_billowed_noise.png')

    return

if __name__ == "__main__":
    main()

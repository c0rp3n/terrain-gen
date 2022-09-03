import numpy as np
import random

from noise import snoise2
from utils import (create_dir, save_heightfield)

DIR = 'heightmaps/'

def main():
    create_dir(DIR)

    hf = np.zeros((512, 512), dtype=np.float32)

    stride = 512 / 4
    for y in range(4):
        for x in range(4):
            hf[y * stride, x * stride] = snoise2(x + 123, y + 521)

    s = 4
    while (s < 512):
        ns = s * 2

        s = ns
        stride = 512 / s

    save_heightfield(hf, DIR + 'hf_subdivision.png')

    return

if __name__ == "__main__":
    main()

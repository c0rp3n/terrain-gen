import numpy as np
import random

from utils import (create_dir, save_heightfield)

DIR = 'heightmaps/'
N = 128

def create_faultline():
    p0 = (random.randrange(0, 512), random.randrange(0, 512))
    p1 = (random.randrange(0, 512), random.randrange(0, 512))

    return p0, p1

def main():
    create_dir(DIR)

    hf = np.ones((512, 512), dtype=np.float32)
    disp = 0.2

    for _ in range(N):
        p0, p1 = create_faultline()
        v = (p1[0] - p0[0], p1[1] - p0[1])

        for index, height in np.ndenumerate(hf):
            x = index[1]
            y = index[0]
            if (v[0] * (y - p0[1])) - (v[1] * (x - p0[0])) > 0:
                hf[index] = height + disp
            else:
                hf[index] = height - disp

        if (disp > 0.05):
            disp = disp * 0.95

    save_heightfield(hf, DIR + 'hf_faulting.png')

    return

if __name__ == "__main__":
    main()

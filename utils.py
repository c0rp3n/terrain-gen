
import numpy as np

from PIL import Image
from os import (mkdir, path)

def create_dir(dir : str) -> None:
    if not path.isdir(dir):
        mkdir(dir)

def save_heightfield(hf : np.ndarray, dir : str) -> None:
    data = np.zeros(hf.shape, dtype=np.uint8)

    hf_min = np.min(hf)
    hf_max = np.max(hf)
    hf_diff = hf_max - hf_min

    for (y, x), height in np.ndenumerate(hf):
        data[y, x] = ((height - hf_min) / hf_diff) * 255

    img = Image.fromarray(data, 'L')
    img.save(dir)

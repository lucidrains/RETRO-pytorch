import numpy as np
from pathlib import Path
from shutil import rmtree
from contextlib import contextmanager

def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors = True)
    path.mkdir(exist_ok = True, parents = True)

@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer

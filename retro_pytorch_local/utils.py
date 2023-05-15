import os
import numpy as np

from pathlib import Path
from shutil import rmtree
from contextlib import contextmanager

def is_true_env_flag(env_flag):
    return os.getenv(env_flag, 'false').lower() in ('true', '1', 't')

def reset_folder_(p):
    path = Path(p)
    rmtree(path, ignore_errors = True)
    path.mkdir(exist_ok = True, parents = True)

@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer

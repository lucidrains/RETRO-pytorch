import os
import numpy as np

from pathlib import Path
from functools import reduce
import math
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


def check_key(key):
    if isinstance(key, tuple):
        raise NotImplementedError("BertEmbeds don't support tuple indexing") 
    if isinstance(key, slice):
        if key.step:
            raise NotImplementedError("BertEmbeds don't support slice steps")
        if key.start < 0 or key.stop < 0:
            raise NotImplementedError("BertEmbeds don't support negative slices")

class BertEmbeds:
    """This class exists solely to workaround the max filesize on ext4
    (16 TB) by pretending to be a single numpy memmap'd array when in
    fact it is two (10TB) split files. If you're here with a file
    bigger than 20 TB, sorry."""

    def __init__(self, fname, dtype, shape, mode):
        elems_per_row =  reduce(lambda acc, x: x * acc, shape[1:])
        bytes_per_row = dtype(0).nbytes * elems_per_row
        print(f'Bytes per row: {bytes_per_row}')
        ten_TB = 1024**4 * 10 

        # We only split when we have more than ten TB of data
        total_bytes = bytes_per_row * shape[0]
        print(f'Total bytes: {total_bytes}')
        self.num_files = math.ceil(total_bytes / ten_TB)
        if self.num_files > 2:
            raise NotImplementedError("BertEmbeds doesn't support files > 20 TB")

        # Each file should have around 10 TB of data. We explicitly do
        # this instead of dividing the rows evenly between files to
        # handle the case when a user constructs this class with a
        # shape other than the original number of rows.
        self.max_rows_per_file = ten_TB // bytes_per_row

        part1_num_rows = shape[0] if self.num_files == 1 else self.max_rows_per_file
        part2_num_rows = shape[0] - self.max_rows_per_file
        self.part1 = np.memmap(fname + '.part1', dtype=dtype, shape=(part1_num_rows, *shape[1:]), mode=mode)
        if part2_num_rows > 0:
            self.part2 = np.memmap(fname + '.part2', dtype=dtype, shape=(part2_num_rows, *shape[1:]), mode=mode)

        self.shape = shape
        self.dtype = dtype

    def __getitem__(self, key):
        check_key(key)
        if isinstance(key, int):
            if key < self.max_rows_per_file:
                return self.part1[key]
            else:
                return self.part2[key - self.max_rows_per_file]

        if isinstance(key, slice):
            if key.stop < self.max_rows_per_file:
                return self.part1[key] 
            elif key.start >= self.max_rows_per_file:
                return self.part2[key.start - self.max_rows_per_file: key.stop - self.max_rows_per_file]
            else:
                return np.concatenate((self.part1[key.start:self.max_rows_per_file], self.part2[:key.stop - self.max_rows_per_file]))

    def __setitem__(self, key, val):
        check_key(key)
        if isinstance(key, int):
            if key < self.max_rows_per_file:
                self.part1[key] = val
            else:
                self.part2[key - self.max_rows_per_file] = val

        if isinstance(key, slice):
            if key.stop < self.max_rows_per_file:
                self.part1[key] = val
            elif key.start >= self.max_rows_per_file:
                self.part2[key.start - self.max_rows_per_file : key.stop - self.max_rows_per_file] = val
            else:
                self.part1[key.start:self.max_rows_per_file] = val[:self.max_rows_per_file - key.start]
                self.part2[:key.stop - self.max_rows_per_file] = val[self.max_rows_per_file - key.start:]

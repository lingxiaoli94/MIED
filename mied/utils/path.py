from pathlib import Path
import numpy as np
import re
import h5py

def grab_step_files(parent_dir, regex='step-([0-9]+)\.(h5|npy)'):
    parent_dir = Path(parent_dir)
    step_files = []
    for f in parent_dir.iterdir():
        m = re.search(regex, str(f))
        if m is not None:
            step_files.append((int(m.group(1)), f))

    step_files.sort(key=lambda pr: pr[0])
    return step_files


def load_samples(file):
    p = Path(file)
    if p.suffix == '.npy':
        return np.load(p)
    else:
        assert(p.suffix == '.h5')
        h5_handle = h5py.File(file, 'r')
        samples = h5_handle['samples'][:]
        h5_handle.close()
        return samples

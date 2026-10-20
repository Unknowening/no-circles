"""Load data_krajbich2010.mat (MATLAB v7.3 / HDF5) into a dict of numpy arrays."""
import h5py
import numpy as np

MAT_FILE = "data_krajbich2010.mat"


def _deref(f, ref):
    return np.array(f[ref]).squeeze()


def load_mat(path=MAT_FILE, include_focus=False):
    """Per-trial (1, N) fields become 1-D arrays of length N.

    The big (20000, N) focus_* arrays are skipped unless include_focus=True.
    `dwells` is returned as lists of per-trial arrays (len, roi).
    """
    data = {}
    with h5py.File(path, "r") as f:
        for key in f:
            if key == "#refs#":
                continue
            obj = f[key]
            if key == "dwells":
                data[key] = {
                    k: [np.atleast_1d(_deref(f, r)) for r in obj[k][:, 0]]
                    for k in obj
                }
            elif key == "filename":
                data[key] = [
                    "".join(chr(c) for c in np.array(f[r]).squeeze()) for r in obj[0]
                ]
            elif key.startswith("focus") and key not in ("focus_last", "focus_prc"):
                if include_focus:
                    data[key] = obj[()].T  # (trials, time)
            else:
                data[key] = obj[()].squeeze()
    return data


if __name__ == "__main__":
    d = load_mat()
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{k:28s} dict of lists, n={len(next(iter(v.values())))}")
        elif isinstance(v, list):
            print(f"{k:28s} list, n={len(v)}")
        else:
            print(f"{k:28s} {v.shape}")
    print("subjects:", np.unique(d["subject_id"]).size, "trials:", d["RT"].size)

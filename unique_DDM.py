"""Load data_krajbich2010.mat (MATLAB v7.3 / HDF5) into a dict of numpy arrays."""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def make_dataframe(d):
    """Trial-level DataFrame in HSSM format: rt, response (+1 = left, -1 = right), dv."""
    df = pd.DataFrame(
        {
            "subject": d["subject_id"].astype(int),
            "rt": d["RT"],
            "response": np.where(d["choice"] == 1, 1, -1),
            "dv": d["dv"],  # vleft - vright
        }
    )
    df["choice_valid"] = ~np.isnan(d["choice"])
    df = df[df.choice_valid & df.rt.notna()].drop(columns="choice_valid")
    return df.reset_index(drop=True)


def fit_ddm(df, draws=1000, tune=1000, chains=4, hierarchical=False):
    """Fit a DDM where drift rate depends linearly on the value difference (dv)."""
    import hssm

    if hierarchical:
        include = [
            {
                "name": "v",
                "formula": "v ~ 0 + dv + (0 + dv | subject)",
            },
            {"name": "a", "formula": "a ~ 1 + (1 | subject)"},
            {"name": "t", "formula": "t ~ 1 + (1 | subject)"},
        ]
    else:
        include = [{"name": "v", "formula": "v ~ 0 + dv"}]

    model = hssm.HSSM(data=df, model="ddm", include=include)
    idata = model.sample(draws=draws, tune=tune, chains=chains, cores=chains)
    return model, idata


if __name__ == "__main__":
    import arviz as az

    d = load_mat()
    df = make_dataframe(d)
    print(f"{df.subject.nunique()} subjects, {len(df)} valid trials")

    model, idata = fit_ddm(df)
    print(az.summary(idata))
    az.plot_trace(idata)
    plt.savefig("ddm_trace.png", dpi=120)
"""Plot per-subject PyDDM parameter estimates from pyddm_fits.csv."""
import matplotlib.pyplot as plt
import pandas as pd

PARAMS = {
    "v": "drift scale (v)",
    "B": "bound (B)",
    "x0": "starting point bias (x0, fraction of B)",
    "nondectime": "non-decision time (s)",
}

if __name__ == "__main__":
    res = pd.read_csv("pyddm_fits.csv").sort_values("subject")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for j, (col, label) in enumerate(PARAMS.items()):
        ax = axes[0, j]
        ax.hist(res[col], bins=12, color="steelblue", edgecolor="white")
        ax.axvline(res[col].median(), color="k", ls="--", lw=1)
        ax.set_title(label)
        ax.set_ylabel("subjects" if j == 0 else "")

        ax = axes[1, j]
        ax.bar(res["subject"].astype(str), res[col], color="steelblue")
        ax.set_xlabel("subject")
        ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig.suptitle("Per-subject DDM parameter estimates (top: distribution, bottom: by subject)")
    fig.tight_layout()
    fig.savefig("pyddm_params.png", dpi=120)

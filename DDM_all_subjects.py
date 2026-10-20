"""Fit an independent value-based drift diffusion model for each participant.

Input: data_krajbich2010.mat in the same folder as this script.
Outputs: DDM_Results.csv and DDM_Parameter_Distributions.png.
"""
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyddm

HERE = Path(__file__).resolve().parent
INPUT = HERE / "data_krajbich2010.mat"
OUTPUT = HERE / "DDM_Results.csv"
FIGURE = HERE / "DDM_Parameter_Distributions.png"

# Read the HDF5-based MATLAB file.
with h5py.File(INPUT, "r") as f:
    df = pd.DataFrame({
        "subject": f["subject_id"][()].ravel(),
        "rt": f["RT"][()].ravel(),
        "choice": f["choice"][()].ravel(),
        "dv": f["dv"][()].ravel(),
    })

# The dataset codes choice=1 as left; choice=0 as right.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["subject", "rt", "choice", "dv"])
df = df[(df["rt"] > 0) & df["choice"].isin([0, 1])].copy()
df["subject"] = df["subject"].astype(int)
df["choice"] = df["choice"].astype(int)

results = []
subject_groups = list(df.groupby("subject", sort=True))
print(f"Fitting {len(subject_groups)} participants / {len(df)} valid trials", flush=True)

for number, (subject, data) in enumerate(subject_groups, start=1):
    # t0 must be smaller than the participant's fastest observed response.
    t0_max = min(1.2, float(data["rt"].min()) - 0.05)
    print(f"[{number}/{len(subject_groups)}] Subject {subject} "
          f"({len(data)} trials)...", flush=True)

    row = {
        "subject": subject,
        "n_trials": len(data),
        "mean_rt": float(data["rt"].mean()),
        "median_rt": float(data["rt"].median()),
        "proportion_left": float(data["choice"].mean()),
        "beta_ddm": np.nan,
        "B": np.nan,
        "t0": np.nan,
        "ddm_implied_softmax_beta": np.nan,
        "nll": np.nan,
        "AIC": np.nan,
        "BIC": np.nan,
        "t0_at_bound": False,
        "any_parameter_at_bound": False,
        "success": False,
        "error": "",
    }

    if t0_max <= 0.01:
        row["error"] = "Minimum reaction time too short for t0 bounds"
        results.append(row)
        continue

    bounds = {"beta": (0, 3), "B": (0.2, 3), "t0": (0, t0_max)}
    try:
        model = pyddm.gddm(
            drift=lambda dv, beta: beta * dv,
            noise=1,
            bound="B",
            nondecision="t0",
            starting_position=0,
            parameters=bounds,
            conditions=["dv"],
            choice_names=("Left", "Right"),
            T_dur=max(16, int(np.ceil(data["rt"].max() + 1))),
        )
        sample = pyddm.Sample.from_pandas_dataframe(
            data[["rt", "choice", "dv"]],
            rt_column_name="rt",
            choice_column_name="choice",
            choice_names=("Left", "Right"),
        )
        model.fit(sample)

        parameter_values = {
            name: float(value)
            for name, value in zip(
                model.get_model_parameter_names(), model.get_model_parameters()
            )
        }
        beta = parameter_values["beta"]
        B = parameter_values["B"]
        t0 = parameter_values["t0"]
        nll = float(model.get_fit_result().value())
        at_bound = {
            key: (abs(parameter_values[key] - lo) < 0.01 or
                  abs(parameter_values[key] - hi) < 0.01)
            for key, (lo, hi) in bounds.items()
        }
        n_params = 3
        row.update({
            "beta_ddm": beta,
            "B": B,
            "t0": t0,
            "ddm_implied_softmax_beta": 2 * B * beta,
            "nll": nll,
            "AIC": 2 * n_params + 2 * nll,
            "BIC": n_params * np.log(len(data)) + 2 * nll,
            "t0_at_bound": at_bound["t0"],
            "any_parameter_at_bound": any(at_bound.values()),
            "success": bool(np.isfinite(nll)),
        })
        print(f"  beta={beta:.3f}  B={B:.3f}  t0={t0:.3f}  "
              f"NLL={nll:.2f}  at_bound={any(at_bound.values())}", flush=True)
    except Exception as exc:
        row["error"] = str(exc)
        print(f"  FIT FAILED: {exc}", flush=True)

    results.append(row)
    # Save after every participant, so earlier results survive interruption.
    pd.DataFrame(results).to_csv(OUTPUT, index=False)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT, index=False)
valid = results_df[results_df["success"]].copy()
print(f"\nFinished: {len(valid)}/{len(results_df)} successful fits")
print(f"Results saved to {OUTPUT}")

if not valid.empty:
    print("\nParameter summary:")
    print(valid[["beta_ddm", "B", "t0"]].describe().round(3))
    print(f"Fits with any parameter near a bound: "
          f"{int(valid['any_parameter_at_bound'].sum())}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (column, label) in zip(axes, [
        ("beta_ddm", "Drift sensitivity (beta)"),
        ("B", "Boundary (B)"),
        ("t0", "Nondecision time (seconds)"),
    ]):
        ax.hist(valid[column], bins=min(12, max(5, len(valid) // 3)), edgecolor="black")
        ax.set_xlabel(label)
        ax.set_ylabel("Participants")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=200)
    plt.close(fig)
    print(f"Figure saved to {FIGURE}")

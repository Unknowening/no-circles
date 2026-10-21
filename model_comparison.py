"""Combine logit and pyddm fits into NLL / AIC / BIC per subject.

logit_fits.csv has nll directly (k=1: beta).
pyddm_fits.csv 'loss' is LossRobustBIC = 2*NLL + k*ln(n), k=4 (v, B, x0, nondectime),
so NLL = (loss - k*ln(n)) / 2.
Note: logit NLL covers choices only; pyddm NLL covers choices AND RTs, so the two
are not directly comparable.
"""
import numpy as np
import pandas as pd

K = {"logit": 1, "pyddm": 4}

logit = pd.read_csv("logit_fits.csv")
ddm = pd.read_csv("pyddm_fits.csv")
ddm["nll"] = (ddm["loss"] - K["pyddm"] * np.log(ddm["n_trials"])) / 2

rows = []
for model, df in [("logit", logit), ("pyddm", ddm)]:
    k = K[model]
    out = df[["subject", "n_trials", "nll"]].copy()
    out.insert(1, "model", model)
    out["k"] = k
    out["aic"] = 2 * k + 2 * out["nll"]
    out["bic"] = k * np.log(out["n_trials"]) + 2 * out["nll"]
    rows.append(out)

res = pd.concat(rows).sort_values(["subject", "model"])
res.to_csv("model_comparison.csv", index=False)
print(res.groupby("model")[["nll", "aic", "bic"]].agg(["mean", "sum"]).round(2))

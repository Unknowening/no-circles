"""Fit a per-subject softmax choice model to data_krajbich2010.mat.

P(choose L) = 1 / (1 + exp(-beta * (r_L - r_R)))

r_L - r_R is the `dv` column (vleft - vright); choice ignores RT.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from unique_DDM import load_mat, make_dataframe


def p_left(beta, dv):
    return 1.0 / (1.0 + np.exp(-beta * dv))


def neg_log_lik(beta, dv, chose_left):
    p = np.clip(p_left(beta, dv), 1e-12, 1 - 1e-12)
    return -np.sum(np.where(chose_left, np.log(p), np.log(1 - p)))


def fit_subject(sub_df):
    dv = sub_df.dv.to_numpy()
    chose_left = (sub_df.response == 1).to_numpy()
    res = minimize_scalar(neg_log_lik, bounds=(-10, 10), args=(dv, chose_left), method="bounded")
    return res.x, res.fun


def fit_all(df):
    rows = []
    for subj, sub_df in df.groupby("subject"):
        beta, nll = fit_subject(sub_df)
        rows.append({"subject": subj, "n_trials": len(sub_df), "beta": beta, "nll": nll})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_dataframe(load_mat())
    print(f"{df.subject.nunique()} subjects, {len(df)} valid trials")
    results = fit_all(df)
    results.to_csv("logit_fits.csv", index=False)
    print(results.describe())

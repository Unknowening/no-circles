"""Fit a separate PyDDM model to each subject in data_krajbich2010.mat.

Drift rate is linear in the value difference (dv = vleft - vright); the upper
boundary is a left choice, the lower boundary a right choice.
"""
import pandas as pd
import pyddm as ddm

from unique_DDM import load_mat, make_dataframe


class DriftDV(ddm.models.Drift):
    """Drift = v * dv, where dv is a per-trial condition."""

    name = "drift proportional to value difference"
    required_parameters = ["v"]
    required_conditions = ["dv"]

    def get_drift(self, conditions, **kwargs):
        return self.v * conditions["dv"]


def build_model():
    return ddm.Model(
        name="ddm_dv",
        drift=DriftDV(v=ddm.Fittable(minval=-1, maxval=1)),
        noise=ddm.NoiseConstant(noise=1),
        bound=ddm.BoundConstant(B=ddm.Fittable(minval=0.3, maxval=4)),
        IC=ddm.ICPointRatio(x0=ddm.Fittable(minval=-0.9, maxval=0.9)),
        overlay=ddm.OverlayNonDecision(nondectime=ddm.Fittable(minval=0, maxval=1.5)),
        dx=0.01,
        dt=0.01,
        T_dur=15,
    )


def fit_subject(sub_df):
    """Fit one subject; returns (fitted model, fit info dict)."""
    sample = ddm.Sample.from_pandas_dataframe(
        sub_df.assign(choice_left=(sub_df.response == 1).astype(int)),
        rt_column_name="rt",
        choice_column_name="choice_left",
    )
    model = build_model()
    ddm.fit_adjust_model(
        sample,
        model,
        fitting_method="differential_evolution",
        lossfunction=ddm.LossRobustBIC,
        verbose=False,
    )
    return model


def fit_all(df):
    rows = []
    for subj, sub_df in df.groupby("subject"):
        model = fit_subject(sub_df)
        params = {
            name: float(p) for name, p in zip(model.get_model_parameter_names(), model.get_model_parameters())
        }
        rows.append(
            {
                "subject": subj,
                "n_trials": len(sub_df),
                **params,
                "loss": float(model.fitresult.value()),
            }
        )
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = make_dataframe(load_mat())
    print(f"{df.subject.nunique()} subjects, {len(df)} valid trials")
    results = fit_all(df)
    results.to_csv("pyddm_fits.csv", index=False)
    print(results.describe())

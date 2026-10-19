
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/shap"
SEED = 00
BACKGROUND_SIZE = 200
ADDIVITY_ATOL = 1e-5


def positive_shap(explanation, class_index=1):
    values = np.asarray(explanation.values)
    base = np.asarray(explanation.base_values)
    if values.ndim == 3:
        values = values[:, :, class_index]
        base = base[:, class_index] if base.ndim == 2 else base[class_index]
    if values.ndim != 2: raise ValueError("error")
    return values, base


def main():
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bundle = joblib.load(ROOT / "results/models/Comp-Trans.joblib")
    model, cols = bundle["model"], bundle["features"]
    clinical = pd.read_csv(ROOT / "input/clinical.csv", dtype={"case_id": str}).set_index("case_id")
    data = pd.read_csv(ROOT / "results/qc/radiomics_scaled.csv", dtype={"case_id": str}).set_index("case_id")
    ids = clinical.index[clinical.cohort == "training"]
    x = data.loc[ids, cols]
    rng = np.random.default_rng(SEED)
    background = x.iloc[rng.choice(len(x), min(BACKGROUND_SIZE, len(x)), replace=False)]
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional",
                                   model_output="probability")
    explanation = explainer(x, check_additivity=True)
    index = list(model.classes_).index(1)
    values, base = positive_shap(explanation, index)
    probability = model.predict_proba(x)[:, index]
    if not np.allclose(np.asarray(base)+values.sum(axis=1), probability, atol=ADDIVITY_ATOL, rtol=1e-5):
        raise ValueError("error")
    importance = pd.DataFrame(dict(feature=cols, mean_abs_shap=np.abs(values).mean(axis=0)))
    importance = importance.sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
    importance["rank"] = importance.mean_abs_shap.rank(method="average", ascending=False)
    metadata = pd.read_csv(ROOT / "results/features/feature_metadata.csv")
    importance = importance.merge(metadata, on="feature", how="left", validate="one_to_one")
    if importance[["source", "compartment", "family"]].isna().any().any():
        raise ValueError("error")
    total = importance.mean_abs_shap.sum()
    if total <= 0: raise ValueError("error")
    importance["sequence"] = importance.source.map(
        {"FS-T2W": "FS-T2W", "ADC": "DW", "MU": "DCE", "TP": "DCE", "UR": "DCE", "WR": "DCE"})
    groups = []
    for grouping in ["source", "sequence", "compartment", "family"]:
        for category, group in importance.groupby(grouping):
            fraction = len(group)/len(cols); contribution = group.mean_abs_shap.sum()/total
            groups.append(dict(grouping=grouping, category=category, n_features=len(group),
                feature_fraction=fraction, importance_fraction=contribution, enrichment=contribution/fraction))
    OUT.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUT / "importance.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT / "group_contributions.csv", index=False)
    rows = pd.DataFrame(dict(case_id=np.repeat(ids.to_numpy(), len(cols)),
        feature=np.tile(cols, len(x)), shap_value=values.ravel(), feature_value=x.to_numpy().ravel()))
    rows.to_csv(OUT / "shap_values.csv", index=False)
    shap.summary_plot(values, x, plot_type="bar", max_display=len(cols), show=False)
    plt.savefig(OUT / "importance.svg", bbox_inches="tight"); plt.close()
    shap.summary_plot(values, x, max_display=len(cols), show=False)
    plt.savefig(OUT / "beeswarm.svg", bbox_inches="tight"); plt.close()
    (OUT / "parameters.json").write_text(json.dumps(dict(seed=SEED,
        background_size=len(background), feature_perturbation="interventional",
        model_output="probability", population="training", SHAP=shap.__version__,
        additivity_max_error=float(np.max(np.abs(np.asarray(base)+values.sum(1)-probability)))), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

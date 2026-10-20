"""
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import norm
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/subgroup_sensitivity"
SEED = 00
N_BOOTSTRAP = 1000
COHORTS = ["training", "external1", "external2", "public"]
SUBTYPES = ["Luminal A/B", "HER2-positive", "Triple-negative"]
SIZE_CUTPOINT_MM = 30.
GRIDS = {"SVM": {"C": [.1, 1., 10.], "gamma": ["scale", .01, .1]},
         "LR": {"C": [.01, .1, 1., 10.]},
         "KNN": {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]},
         "DT": {"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5]}}


def load_performance():
    spec = importlib.util.spec_from_file_location("model_performance", ROOT / "11_model_performance.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def classifier(name, params):
    if name == "SVM": model = SVC(**params, probability=True, random_state=SEED)
    elif name == "LR": model = LogisticRegression(**params, max_iter=10000, random_state=SEED)
    elif name == "KNN": model = KNeighborsClassifier(**params)
    elif name == "DT": model = DecisionTreeClassifier(**params, random_state=SEED)
    else: raise ValueError(name)
    return make_pipeline(StandardScaler(), model)


def independent_auc_test(y1, p1, y2, p2, perf):
    a1, a0, auc1 = perf.delong_components(y1, p1)
    b1, b0, auc2 = perf.delong_components(y2, p2)
    variance = np.var(a1, ddof=1)/len(a1)+np.var(a0, ddof=1)/len(a0)
    variance += np.var(b1, ddof=1)/len(b1)+np.var(b0, ddof=1)/len(b0)
    diff = auc1-auc2
    p = 1. if variance == 0 and diff == 0 else (0. if variance == 0 else float(2*norm.sf(abs(diff)/np.sqrt(variance))))
    return float(diff), p


def main():
    from itertools import combinations
    perf = load_performance()
    clinical = pd.read_csv(ROOT / "input/clinical.csv", dtype={"case_id": str}).set_index("case_id")
    predictions = pd.read_csv(ROOT / "results/models/predictions.csv", dtype={"case_id": str}).set_index("case_id")
    data = clinical.join(predictions[["Comp-Trans"]], validate="one_to_one")
    if data[["molecular_subtype", "tumor_diameter_mm"]].isna().any().any():
        raise ValueError("error")
    external = data[data.cohort.isin(COHORTS[1:])].copy()
    if not external.molecular_subtype.isin(SUBTYPES).all(): raise ValueError("error")
    external["size_group"] = np.where(external.tumor_diameter_mm <= SIZE_CUTPOINT_MM, "<30mm", ">=30mm")
    thresholds = json.loads((ROOT / "results/models/thresholds.json").read_text())
    rng = np.random.default_rng(SEED); rows = []; comparisons = []; curve_rows = []
    for grouping, levels in [("molecular_subtype", SUBTYPES), ("size_group", ["<30mm", ">=30mm"])]:
        groups = {}
        for level in levels:
            group = external[external[grouping] == level]
            y, p = group.pCR.to_numpy(int), group["Comp-Trans"].to_numpy(float)
            if set(y) != {0, 1}: raise ValueError(f"{level}error")
            groups[level] = (y, p)
            point = perf.classification_metrics(y, p, thresholds["Comp-Trans"])
            boots = [perf.classification_metrics(y[idx], p[idx], thresholds["Comp-Trans"])
                     for idx in [perf.stratified_indices(y, rng) for _ in range(N_BOOTSTRAP)]]
            for metric in ["AUC", "ACC", "SEN", "SPE", "PPV", "NPV"]:
                lo, hi = perf.interval([b[metric] for b in boots])
                rows.append(dict(grouping=grouping, subgroup=level, metric=metric, estimate=point[metric],
                    lower=lo, upper=hi, n=len(y), **{c: point[c] for c in ["TP", "TN", "FP", "FN"]}))
            fpr, tpr, _ = roc_curve(y, p)
            curve_rows.extend(dict(grouping=grouping, subgroup=level, x=a, y=b) for a, b in zip(fpr, tpr))
        family = []
        for a, b in combinations(levels, 2):
            diff, pv = independent_auc_test(*groups[a], *groups[b], perf)
            family.append(dict(grouping=grouping, subgroup_a=a, subgroup_b=b, auc_difference=diff, p=pv))
        q = perf.bh([r["p"] for r in family])
        comparisons.extend(dict(**r, q=float(v)) for r, v in zip(family, q))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "subgroup_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "subgroup_comparisons.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "subgroup_curves.csv", index=False)
    features = pd.read_csv(ROOT / "results/qc/radiomics_scaled.csv", dtype={"case_id": str}).set_index("case_id")
    bundle = joblib.load(ROOT / "results/models/Comp-Trans.joblib"); cols = bundle["features"]
    train, tune = data.index[data.cohort == "training"], data.index[data.cohort == "tuning"]
    sens = data[["cohort", "pCR"]].copy(); sens["RF"] = data["Comp-Trans"]
    sens_thresholds = {"RF": thresholds["Comp-Trans"]}; grid_rows = []
    for name, grid in GRIDS.items():
        best = None
        for parameters in ParameterGrid(grid):
            model = classifier(name, parameters).fit(features.loc[train, cols], data.loc[train, "pCR"])
            auc = roc_auc_score(data.loc[tune, "pCR"], model.predict_proba(features.loc[tune, cols])[:, 1])
            grid_rows.append(dict(classifier=name, tuning_auc=float(auc), parameters=json.dumps(parameters)))
            if best is None or auc > best[0]: best = (auc, parameters, model)
        sens[name] = best[2].predict_proba(features.loc[data.index, cols])[:, 1]
        fpr, tpr, cuts = roc_curve(data.loc[train, "pCR"], sens.loc[train, name])
        valid = np.isfinite(cuts) & (cuts >= 0) & (cuts <= 1)
        jj = tpr[valid]-fpr[valid]
        sens_thresholds[name] = float(np.max(cuts[valid][np.isclose(jj, jj.max())]))
        joblib.dump(dict(model=best[2], features=cols, threshold=sens_thresholds[name],
                         parameters=best[1]), OUT / f"{name}.joblib")
    summary = []; sens_curves = []
    for cohort in COHORTS:
        group = sens[sens.cohort == cohort]; y = group.pCR.to_numpy(int)
        indices = [perf.stratified_indices(y, rng) for _ in range(N_BOOTSTRAP)]
        for name in ["RF", *GRIDS]:
            p = group[name].to_numpy(float)
            auc = roc_auc_score(y, p)
            lo, hi = perf.interval([roc_auc_score(y[i], p[i]) for i in indices])
            summary.append(dict(cohort=cohort, classifier=name, AUC=auc, lower=lo, upper=hi))
            fpr, tpr, _ = roc_curve(y, p)
            sens_curves.extend(dict(cohort=cohort, classifier=name, x=a, y=b) for a, b in zip(fpr, tpr))
    sens.to_csv(OUT / "classifier_predictions.csv", index_label="case_id")
    pd.DataFrame(summary).to_csv(OUT / "classifier_auc.csv", index=False)
    pd.DataFrame(sens_curves).to_csv(OUT / "classifier_curves.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(OUT / "classifier_tuning.csv", index=False)
    (OUT / "parameters.json").write_text(json.dumps(dict(size_cutpoint_mm=SIZE_CUTPOINT_MM,
        subtype_levels=SUBTYPES, bootstrap=N_BOOTSTRAP, seed=SEED, grids=GRIDS,
        class_weights="RF balanced; other estimators default", external_refit=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

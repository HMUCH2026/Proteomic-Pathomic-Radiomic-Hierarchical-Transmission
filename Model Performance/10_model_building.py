
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/models"
SEED = 00
GRID = {"n_estimators": [300, 600], "max_depth": [None, 5, 10],
        "max_features": ["sqrt", .5], "min_samples_leaf": [1, 3, 5]}
N_JOBS = 1


def training_threshold(y, probability):
    fpr, tpr, thresholds = roc_curve(y, probability)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    if not valid.any(): raise ValueError("error")
    j = tpr[valid]-fpr[valid]
    
    return float(np.max(thresholds[valid][np.isclose(j, j.max())]))


def main():
    features = pd.read_csv(ROOT / "results/qc/radiomics_scaled.csv", dtype={"case_id": str}).set_index("case_id")
    clinical = pd.read_csv(ROOT / "input/clinical.csv", dtype={"case_id": str}).set_index("case_id")
    sets = json.loads((ROOT / "results/selection/final_sets.json").read_text())
    if set(sets) != {"Comp-Trans", "Part-Trans", "Direct-Dom", "Trans-Null", "Data-Driven"}:
        raise ValueError("error")
    if clinical.index.duplicated().any() or not clinical.index.isin(features.index).all():
        raise ValueError("error")
    train = clinical.index[clinical.cohort == "training"]
    tune = clinical.index[clinical.cohort == "tuning"]
    if not len(train) or not len(tune): raise ValueError("error")
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = clinical[["cohort", "pCR"]].copy(); thresholds = {}; search = []
    for name, cols in sets.items():
        xtrain, xtune = features.loc[train, cols], features.loc[tune, cols]
        if not np.isfinite(features.loc[clinical.index, cols].to_numpy()).all():
            raise ValueError(f"{name}: error")
        best = None
        for index, params in enumerate(ParameterGrid(GRID)):
            rf = RandomForestClassifier(**params, random_state=SEED,
                class_weight="balanced", n_jobs=N_JOBS).fit(xtrain, clinical.loc[train, "pCR"])
            p = rf.predict_proba(xtune)[:, list(rf.classes_).index(1)]
            score = float(roc_auc_score(clinical.loc[tune, "pCR"], p))
            search.append(dict(model=name, grid_index=index, tuning_auc=score, **params))
            
            if best is None or score > best[0]: best = (score, params)
        rf = RandomForestClassifier(**best[1], random_state=SEED,
            class_weight="balanced", n_jobs=N_JOBS).fit(xtrain, clinical.loc[train, "pCR"])
        probability = rf.predict_proba(features.loc[clinical.index, cols])[:, list(rf.classes_).index(1)]
        predictions[name] = probability
        thresholds[name] = training_threshold(clinical.loc[train, "pCR"], predictions.loc[train, name])
        joblib.dump(dict(model=rf, features=cols, threshold=thresholds[name],
            tuned_parameters=best[1], training_cohort="training", outcome_positive=1,
            preprocessing="R03 training-fitted transform; supplied scaled matrix"), OUT / f"{name}.joblib")
    predictions.to_csv(OUT / "predictions.csv", index_label="case_id")
    pd.DataFrame(search).to_csv(OUT / "hyperparameter_search.csv", index=False)
    (OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

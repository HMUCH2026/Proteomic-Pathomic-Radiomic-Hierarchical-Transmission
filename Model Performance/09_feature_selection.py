
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/selection"
FEATURES = ROOT / "results/qc/radiomics_scaled.csv"
CLINICAL = ROOT / "input/clinical.csv"
TYPES = ["Comp-Trans", "Part-Trans", "Direct-Dom", "Trans-Null"]
SEED = 00
Q = .05
VIF_MAX = 5.
CORR_MAX = .80
N_SPLITS, N_REPEATS = 5, 10
C_GRID = np.logspace(-4, 2, 25)  
LASSO_MAX_ITER = 20000
BORUTA_MAX_ITER = 100
BORUTA_INCLUDE_TENTATIVE = False  
COUNT_RF_PARAMETERS = dict(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                           class_weight="balanced", random_state=SEED, n_jobs=1)


def bh(p):
    p = np.asarray(p, float); idx = np.argsort(p); q = np.empty(len(p))
    q[idx] = np.minimum(1, np.minimum.accumulate(
        (p[idx]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]); return q


def vifs(x):
    x = np.asarray(x, float); vals = []
    for j in range(x.shape[1]):
        y = x[:, j]; d = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        residual = y-d@np.linalg.lstsq(d, y, rcond=None)[0]
        denominator = np.sum((y-y.mean())**2)
        ratio = np.sum(residual**2)/denominator if denominator > 0 else 0
        vals.append(np.inf if ratio <= 1e-12 else 1/ratio)
    return np.asarray(vals)


def screen_pool(x, y):
    records = []
    for name in x:
        p = mannwhitneyu(x.loc[y == 1, name], x.loc[y == 0, name],
                         alternative="two-sided", method="auto").pvalue
        records.append(dict(feature=name, p=float(p)))
    stats = pd.DataFrame(records)
    if stats.empty: raise ValueError("error")
    stats["q"] = bh(stats.p)
    cols = stats.loc[stats.q < Q, "feature"].tolist()
    if not cols: raise ValueError("error")
    pmap = stats.set_index("feature").p.to_dict()
    while len(cols) > 1:
        scores = vifs(x[cols])
        if scores.max() < VIF_MAX: break
        worst = max(range(len(cols)), key=lambda j: (scores[j], pmap[cols[j]], cols[j]))
        cols.pop(worst)
    while len(cols) > 1:
        corr = x[cols].corr().abs().to_numpy()
        pairs = np.argwhere(np.triu(corr > CORR_MAX, 1))
        if not len(pairs): break
        a, b = pairs[0]; fa, fb = cols[a], cols[b]
        remove = fa if (pmap[fa], fa) > (pmap[fb], fb) else fb
        cols.remove(remove)
    return cols, stats


def logistic_lasso_ranking(x, y, splits, n_splits=N_SPLITS):
    x = np.asarray(x, float); y = np.asarray(y, int)
    losses = np.empty((len(C_GRID), len(splits)))
    for ci, penalty_c in enumerate(C_GRID):
        for fi, (train, valid) in enumerate(splits):
            scaler = StandardScaler().fit(x[train])
            classifier = LogisticRegression(C=float(penalty_c), penalty="l1", solver="liblinear",
                max_iter=LASSO_MAX_ITER, random_state=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(scaler.transform(x[train]), y[train])
            losses[ci, fi] = log_loss(y[valid], classifier.predict_proba(
                scaler.transform(x[valid]))[:, 1], labels=[0, 1])
    repeat_means = losses.reshape(len(C_GRID), -1, n_splits).mean(axis=2)
    means = repeat_means.mean(axis=1)
    se = repeat_means.std(axis=1, ddof=1)/np.sqrt(repeat_means.shape[1])
    best = int(np.argmin(means)); eligible = np.flatnonzero(means <= means[best]+se[best])
    chosen = int(eligible[0])  
    scaler = StandardScaler().fit(x)
    final = LogisticRegression(C=float(C_GRID[chosen]), penalty="l1", solver="liblinear",
        max_iter=LASSO_MAX_ITER, random_state=SEED).fit(scaler.transform(x), y)
    return final.coef_.ravel(), pd.DataFrame(dict(C=C_GRID, mean_log_loss=means,
        standard_error=se, chosen=np.arange(len(C_GRID)) == chosen))


def main():
    from boruta import BorutaPy
    features = pd.read_csv(FEATURES, dtype={"case_id": str}).set_index("case_id")
    clinical = pd.read_csv(CLINICAL, dtype={"case_id": str}).set_index("case_id")
    train_ids = clinical.index[clinical.cohort == "training"]
    tune_ids = clinical.index[clinical.cohort == "tuning"]
    if not len(tune_ids) or not train_ids.isin(features.index).all() or not tune_ids.isin(features.index).all():
        raise ValueError("error")
    x = features.loc[train_ids]; y = clinical.loc[train_ids, "pCR"].astype(int)
    if set(y) != {0, 1} or min(y.value_counts()) < N_SPLITS:
        raise ValueError("error")
    if not np.isfinite(x.to_numpy(float)).all(): raise ValueError("error")
    types = pd.read_csv(ROOT / "results/transmission/transmission_features.csv")
    pools = {t: types.loc[types.type == t, "feature"].tolist() for t in TYPES}
    base = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=1)
    boruta = BorutaPy(base, n_estimators="auto", max_iter=BORUTA_MAX_ITER,
                      random_state=SEED, alpha=Q).fit(x.to_numpy(float), y.to_numpy())
    support = boruta.support_.copy()
    if BORUTA_INCLUDE_TENTATIVE: support |= boruta.support_weak_
    pools["Data-Driven"] = x.columns[support].tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    cv = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                    random_state=SEED).split(x, y))
    rankings = {}; screened = {}
    for name, candidates in pools.items():
        if not set(candidates) <= set(x): raise ValueError(f"{name}: error")
        kept, stats = screen_pool(x[candidates], y)
        coef, cvstats = logistic_lasso_ranking(x[kept], y, cv)
        ranked = pd.DataFrame(dict(feature=kept, coefficient=coef))
        ranked = ranked[ranked.coefficient != 0].assign(
            absolute=lambda d: d.coefficient.abs()).sort_values(["absolute", "feature"], ascending=[False, True])
        if ranked.empty: raise ValueError(f"{name}: error")
        rankings[name] = ranked.feature.tolist(); screened[name] = kept
        stats.to_csv(OUT / f"{name}_univariable.csv", index=False)
        cvstats.to_csv(OUT / f"{name}_lasso_cv.csv", index=False)
        ranked.to_csv(OUT / f"{name}_ranking.csv", index=False)
    count_results = []
    for k in range(1, len(rankings["Data-Driven"])+1):
        cols = rankings["Data-Driven"][:k]
        rf = RandomForestClassifier(**COUNT_RF_PARAMETERS).fit(x[cols], y)
        auc = roc_auc_score(clinical.loc[tune_ids, "pCR"],
            rf.predict_proba(features.loc[tune_ids, cols])[:, 1])
        count_results.append(dict(n_features=k, tuning_auc=float(auc)))
    count_table = pd.DataFrame(count_results)
    best_k = int(count_table.sort_values(["tuning_auc", "n_features"], ascending=[False, True]).iloc[0].n_features)
    if any(len(v) < best_k for v in rankings.values()):
        raise ValueError("error")
    final_sets = {name: cols[:best_k] for name, cols in rankings.items()}
    count_table.to_csv(OUT / "feature_count_tuning.csv", index=False)
    (OUT / "final_sets.json").write_text(json.dumps(final_sets, indent=2), encoding="utf-8")
    (OUT / "candidate_pools.json").write_text(json.dumps(pools, indent=2), encoding="utf-8")
    (OUT / "parameters.json").write_text(json.dumps(dict(seed=SEED,
        folds=N_SPLITS, repeats=N_REPEATS, one_se_unit="repeat_mean", C_grid=C_GRID.tolist(),
        boruta_include_tentative=BORUTA_INCLUDE_TENTATIVE, common_count=best_k,
        feature_count_rf=COUNT_RF_PARAMETERS, preselection="fixed_before_cv"), indent=2), encoding="utf-8")
    pd.DataFrame([dict(repeat=i//N_SPLITS, fold=i%N_SPLITS, case_id=x.index[j])
                  for i, (_, valid) in enumerate(cv) for j in valid]).to_csv(
                      OUT / "cv_partitions.csv", index=False)


if __name__ == "__main__":
    main()

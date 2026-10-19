
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kendalltau
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/replication"
COHORTS = ["external1", "external2", "public"]
ORDER = ["Trans-Null", "Direct-Dom", "Part-Trans", "Comp-Trans"]
N_BOOTSTRAP = 2000
SEED = 00
Q = .05


def bh(p):
    p = np.asarray(p, float); order = np.argsort(p); result = np.empty(len(p))
    result[order] = np.minimum(1, np.minimum.accumulate(
        (p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]); return result


def feature_replication(x, y, sets, train_directions):
    union = sorted(set(f for values in sets.values() for f in values))
    pvalues = {}; directions = {}
    for f in union:
        a, b = x.loc[y == 1, f].to_numpy(), x.loc[y == 0, f].to_numpy()
        pvalues[f] = float(mannwhitneyu(a, b, alternative="two-sided", method="auto").pvalue)
        directions[f] = float(np.sign(np.median(a)-np.median(b)))
    rows = []
    for name, features in sets.items():
        q = bh([pvalues[f] for f in features])
        for f, adj in zip(features, q):
            replicate = bool(adj < Q and train_directions[f] != 0 and directions[f] == train_directions[f])
            rows.append(dict(model=name, feature=f, p=pvalues[f], q=adj,
                direction=directions[f], training_direction=train_directions[f], replicated=replicate))
    return pd.DataFrame(rows)


def summarize(x, y, probabilities, sets, directions):
    details = feature_replication(x.reset_index(drop=True), np.asarray(y), sets, directions)
    rows = []
    for name in sets:
        r = details[details.model == name]
        rows.append(dict(model=name, replication_rate=r.replicated.mean(),
                         AUC=float(roc_auc_score(y, probabilities[name]))))
    return pd.DataFrame(rows), details


def contrasts_and_trends(rows):
    result = {}
    four = rows[rows.model.isin(ORDER)].copy()

    c = pd.get_dummies(pd.Categorical(four.cohort, categories=COHORTS), drop_first=True, dtype=float)
    design = np.column_stack([np.ones(len(four)), c.to_numpy(), four.model.map(dict(zip(ORDER, range(4))))])
    for metric in ["replication_rate", "AUC"]:
        result["trend_"+metric] = float(np.linalg.lstsq(design, four[metric], rcond=None)[0][-1])
    for cohort in COHORTS:
        g = rows[rows.cohort == cohort].set_index("model")
        result[cohort+"_tau_b"] = float(kendalltau(g.loc[ORDER, "replication_rate"],
                                                   g.loc[ORDER, "AUC"], variant="b").statistic)
        for metric in ["replication_rate", "AUC"]:
            result[cohort+"_delta_"+metric] = float(g.loc["Comp-Trans", metric]-g.loc["Data-Driven", metric])
    return result


def main():
    x = pd.read_csv(ROOT / "results/qc/radiomics_scaled.csv", dtype={"case_id": str}).set_index("case_id")
    predictions = pd.read_csv(ROOT / "results/models/predictions.csv", dtype={"case_id": str}).set_index("case_id")
    sets = json.loads((ROOT / "results/selection/final_sets.json").read_text())
    train = predictions[predictions.cohort == "training"]
    union = sorted(set(f for cols in sets.values() for f in cols))
    training = x.loc[train.index]
    directions = {f: float(np.sign(training.loc[train.pCR == 1, f].median()-
                                    training.loc[train.pCR == 0, f].median())) for f in union}
    groups = {}; point_rows = []; details = []
    for cohort in COHORTS:
        g = predictions[predictions.cohort == cohort]
        if set(g.pCR) != {0, 1}: raise ValueError(f"{cohort}error")
        f = x.loc[g.index, union].reset_index(drop=True)
        if not np.isfinite(f.to_numpy()).all(): raise ValueError("error")
        y = g.pCR.to_numpy(int); probability = g[list(sets)].reset_index(drop=True)
        groups[cohort] = (f, y, probability)
        row, detail = summarize(f, y, probability, sets, directions)
        row["cohort"] = cohort; detail["cohort"] = cohort
        point_rows.append(row); details.append(detail)
    point_rows = pd.concat(point_rows, ignore_index=True)
    point = contrasts_and_trends(point_rows)
    rng = np.random.default_rng(SEED); boot_summaries = []; boot_metrics = []
    for b in range(N_BOOTSTRAP):
        rows = []
        for cohort, (f, y, probability) in groups.items():
            index = np.concatenate([rng.choice(np.flatnonzero(y == k), np.sum(y == k), replace=True)
                                    for k in [0, 1]])
            s, _ = summarize(f.iloc[index].reset_index(drop=True), y[index],
                             probability.iloc[index].reset_index(drop=True), sets, directions)
            s["cohort"] = cohort; rows.append(s)
        combined = pd.concat(rows, ignore_index=True)
        values = contrasts_and_trends(combined); values["bootstrap"] = b
        boot_metrics.append(values)
        combined["bootstrap"] = b; boot_summaries.append(combined)
    boots = pd.DataFrame(boot_metrics); summarized = []
    for name, estimate in point.items():
        values = boots[name].to_numpy(); bad = int(np.sum(~np.isfinite(values)))
        lo, hi = np.quantile(values, [.025, .975]) if not bad else (np.nan, np.nan)
        summarized.append(dict(statistic=name, estimate=estimate, lower=lo, upper=hi,
                               undefined_bootstraps=bad))
    all_boot = pd.concat(boot_summaries, ignore_index=True)
    rate_ci = []
    for (cohort, model), g in all_boot.groupby(["cohort", "model"]):
        for metric in ["replication_rate", "AUC"]:
            lo, hi = g[metric].quantile([.025, .975]).to_numpy()
            estimate = point_rows[(point_rows.cohort == cohort) & (point_rows.model == model)][metric].iloc[0]
            rate_ci.append(dict(cohort=cohort, model=model, metric=metric, estimate=estimate, lower=lo, upper=hi))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(details).to_csv(OUT / "feature_replication.csv", index=False)
    point_rows.to_csv(OUT / "set_replication_auc.csv", index=False)
    pd.DataFrame(rate_ci).to_csv(OUT / "set_intervals.csv", index=False)
    pd.DataFrame(summarized).to_csv(OUT / "trends_concordance_contrasts.csv", index=False)
    boots.to_csv(OUT / "bootstrap_statistics.csv", index=False)
    (OUT / "parameters.json").write_text(json.dumps(dict(bootstrap=N_BOOTSTRAP, seed=SEED,
        order=ORDER, order_scores=list(range(4)), q=Q, models_refitted=False,
        ci="patient-level stratified percentile; undefined tau retained"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

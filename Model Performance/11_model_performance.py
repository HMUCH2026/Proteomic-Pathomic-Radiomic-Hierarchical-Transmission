
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/performance"
PREDICTIONS = ROOT / "results/models/predictions.csv"
MODELS = ["Comp-Trans", "Part-Trans", "Direct-Dom", "Trans-Null", "Data-Driven"]
COHORTS = ["training", "external1", "external2", "public"]
N_BOOTSTRAP = 1000
SEED = 00
NRI_MODE = None  
NRI_CUTPOINTS = None  
DCA_THRESHOLDS = np.linspace(.01, .99, 99)
CALIBRATION_BINS = 10


def bh(p):
    p = np.asarray(p, float); index = np.argsort(p); out = np.empty(len(p))
    out[index] = np.minimum(1, np.minimum.accumulate(
        (p[index]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]); return out


def stratified_indices(y, rng):
    return np.concatenate([rng.choice(np.flatnonzero(y == cls), np.sum(y == cls), replace=True)
                           for cls in [0, 1]])


def classification_metrics(y, probability, threshold):
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    pred = probability >= threshold
    tp = int(np.sum(pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    ratio = lambda a, b: a/b if b else np.nan
    return dict(AUC=roc_auc_score(y, probability), ACC=(tp+tn)/len(y),
        SEN=ratio(tp, tp+fn), SPE=ratio(tn, tn+fp), PPV=ratio(tp, tp+fp),
        NPV=ratio(tn, tn+fn), AP=average_precision_score(y, probability),
        Brier=float(np.mean((probability-y)**2)), TP=tp, TN=tn, FP=fp, FN=fn)


def delong_components(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    positive, negative = probability[y == 1], probability[y == 0]
    if min(len(positive), len(negative)) < 2: raise ValueError("error")
    kernel = (positive[:, None] > negative[None, :]).astype(float)
    kernel += .5*(positive[:, None] == negative[None, :])
    return kernel.mean(1), kernel.mean(0), float(kernel.mean())


def paired_delong(y, reference, candidate):
    a1, a0, auc_a = delong_components(y, reference)
    b1, b0, auc_b = delong_components(y, candidate)
    difference = auc_b-auc_a
    variance = np.var(b1-a1, ddof=1)/len(a1)+np.var(b0-a0, ddof=1)/len(a0)
    p = 1. if variance == 0 and difference == 0 else (0. if variance == 0 else
        float(2*norm.sf(abs(difference)/np.sqrt(variance))))
    return difference, p


def reclassification(y, reference, candidate, mode, cutpoints=None):
    y = np.asarray(y); reference, candidate = np.asarray(reference), np.asarray(candidate)
    if mode == "continuous":
        change = np.sign(candidate-reference)
    elif mode == "categorical":
        if cutpoints is None or not len(cutpoints): raise ValueError("error")
        change = np.sign(np.digitize(candidate, cutpoints)-np.digitize(reference, cutpoints))
    else:
        raise ValueError("error")
    nri = float(change[y == 1].mean()-change[y == 0].mean())
    delta = candidate-reference
    idi = float(delta[y == 1].mean()-delta[y == 0].mean())
    return nri, idi


def interval(values):
    a = np.asarray(values, float)
    
    return np.quantile(a, [.025, .975]) if np.isfinite(a).all() else (np.nan, np.nan)


def wald_bootstrap_p(estimate, replicates):
    se = float(np.std(replicates, ddof=1))
    return 1. if se == 0 and estimate == 0 else (0. if se == 0 else float(2*norm.sf(abs(estimate/se))))


def main():
    if NRI_MODE not in {"continuous", "categorical"}:
        raise ValueError("error")
    data = pd.read_csv(PREDICTIONS, dtype={"case_id": str})
    thresholds = json.loads((ROOT / "results/models/thresholds.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, comparisons, curves, calibrations, dca_rows = [], [], [], [], []
    for cohort in COHORTS:
        group = data[data.cohort == cohort]
        y = group.pCR.to_numpy(int); p = group[MODELS].to_numpy(float)
        if not len(y) or set(y) != {0, 1} or not np.isfinite(p).all():
            raise ValueError(f"{cohort}: error")
        if np.any((p < 0) | (p > 1)): raise ValueError("error")
        point = [classification_metrics(y, p[:, j], thresholds[name]) for j, name in enumerate(MODELS)]
        names = ["AUC", "ACC", "SEN", "SPE", "PPV", "NPV", "AP", "Brier"]
        boot = np.empty((N_BOOTSTRAP, len(MODELS), len(names)))
        reclass_boot = np.empty((N_BOOTSTRAP, len(MODELS)-1, 2))
        rng = np.random.default_rng(SEED+COHORTS.index(cohort))
        for b in range(N_BOOTSTRAP):
            idx = stratified_indices(y, rng)
            for j, name in enumerate(MODELS):
                value = classification_metrics(y[idx], p[idx, j], thresholds[name])
                boot[b, j] = [value[k] for k in names]
                if j:
                    reclass_boot[b, j-1] = reclassification(y[idx], p[idx, 0], p[idx, j],
                                                          NRI_MODE, NRI_CUTPOINTS)
        for j, model in enumerate(MODELS):
            for k, metric in enumerate(names):
                lo, hi = interval(boot[:, j, k])
                metrics.append(dict(cohort=cohort, model=model, metric=metric,
                    estimate=point[j][metric], lower=lo, upper=hi, n=len(y),
                    threshold=thresholds[model], undefined_bootstraps=int(np.sum(~np.isfinite(boot[:, j, k]))),
                    **{c: point[j][c] for c in ["TP", "TN", "FP", "FN"]}))
            fpr, tpr, _ = roc_curve(y, p[:, j])
            precision, recall, _ = precision_recall_curve(y, p[:, j])
            curves.extend(dict(cohort=cohort, model=model, curve="ROC", x=float(a), y=float(b))
                          for a, b in zip(fpr, tpr))
            curves.extend(dict(cohort=cohort, model=model, curve="PR", x=float(a), y=float(b))
                          for a, b in zip(recall, precision))
            bins = pd.qcut(pd.Series(p[:, j]), CALIBRATION_BINS, duplicates="drop")
            cal = pd.DataFrame(dict(probability=p[:, j], y=y, bin=bins))
            for _, cell in cal.groupby("bin", observed=True):
                calibrations.append(dict(cohort=cohort, model=model, predicted=cell.probability.mean(),
                                         observed=cell.y.mean(), n=len(cell)))
            for threshold in DCA_THRESHOLDS:
                positive = p[:, j] >= threshold
                net = (np.sum(positive & (y == 1))-np.sum(positive & (y == 0))*threshold/(1-threshold))/len(y)
                dca_rows.append(dict(cohort=cohort, model=model, threshold=threshold, net_benefit=net,
                    treat_all=float(y.mean()-(1-y.mean())*threshold/(1-threshold)), treat_none=0.))
        cohort_comparisons = []
        for j, model in enumerate(MODELS[1:], 1):
            delta, p_auc = paired_delong(y, p[:, 0], p[:, j])
            vals = reclassification(y, p[:, 0], p[:, j], NRI_MODE, NRI_CUTPOINTS)
            for metric, value, replicates, pvalue in [
                ("delta_AUC", delta, boot[:, j, 0]-boot[:, 0, 0], p_auc),
                ("NRI", vals[0], reclass_boot[:, j-1, 0], wald_bootstrap_p(vals[0], reclass_boot[:, j-1, 0])),
                ("IDI", vals[1], reclass_boot[:, j-1, 1], wald_bootstrap_p(vals[1], reclass_boot[:, j-1, 1]))]:
                lo, hi = interval(replicates)
                cohort_comparisons.append(dict(cohort=cohort, model=model, reference=MODELS[0],
                    metric=metric, estimate=value, lower=lo, upper=hi, p=pvalue,
                    direction="candidate-minus-Comp-Trans"))
        comparison = pd.DataFrame(cohort_comparisons)
        comparison["q"] = comparison.groupby("metric")["p"].transform(lambda s: bh(s.to_numpy()))
        comparisons.extend(comparison.to_dict("records"))
    for filename, rows in [("metrics.csv", metrics), ("comparisons.csv", comparisons),
                           ("curves.csv", curves), ("calibration.csv", calibrations), ("dca.csv", dca_rows)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False)
    (OUT / "parameters.json").write_text(json.dumps(dict(bootstrap=N_BOOTSTRAP, seed=SEED,
        stratification="pCR", nri=NRI_MODE, nri_cutpoints=NRI_CUTPOINTS,
        nri_idi_p="two-sided Wald estimate/bootstrap-SE", pr_area="average_precision",
        CI="patient-level stratified percentile", thresholds="frozen_training_Youden"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

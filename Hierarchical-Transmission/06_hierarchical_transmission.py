from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/transmission"
CLINICAL = ROOT / "input/clinical.csv"
X_FILE = ROOT / "results/pathways/training_scores.csv"
M_FILE = ROOT / "results/qc/pathomics_scaled.csv"
Y_FILE = ROOT / "results/qc/radiomics_scaled.csv"
CLINICAL_COLS = ["age", "ER", "PR", "HER2", "Ki67"]
Q_THRESHOLD = .05
VIF_THRESHOLD = 5.
N_BOOTSTRAP = 2000
SEED = 00


def bh(p):
    a = np.asarray(p, float)
    if not np.isfinite(a).all(): raise ValueError("error")
    order = np.argsort(a); n = len(a)
    q = np.empty(n); q[order] = np.minimum(1, np.minimum.accumulate(
        (a[order]*n/np.arange(1, n+1))[::-1])[::-1])
    return q


def frame(path):
    d = pd.read_csv(path, dtype={"case_id": str})
    if d.case_id.duplicated().any(): raise ValueError("error")
    return d.set_index("case_id")


def require_finite(*arrays):
    if any(not np.isfinite(np.asarray(a, float)).all() for a in arrays):
        raise ValueError("error")


def design(*arrays):
    n = len(arrays[0]); x = np.column_stack([np.ones(n), *arrays])
    require_finite(x)
    if len(x) <= x.shape[1] or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("error")
    return x


def r2_fit(y, d):
    b, _, _, _ = np.linalg.lstsq(d, y, rcond=None)
    residual = y-d@b
    sse = np.sum(residual**2, axis=0)
    sst = np.sum((y-y.mean(axis=0))**2, axis=0)
    if np.any(sst <= 0): raise ValueError("error")
    return 1-sse/sst, sse


def variance_components(x, m, c, y):
    y = np.asarray(y, float)
    if y.ndim == 1: y = y[:, None]
    require_finite(x, m, c, y)
    designs = [design(c), design(c, x), design(c, m), design(c, x, m)]
    fits = [r2_fit(y, d) for d in designs]
    rc, rx, rm, rxm = [v[0] for v in fits]
    overall, direct = rx-rc, rxm-rm
    transmitted = overall-direct
    tests = []
    for reduced, full in [(0, 1), (2, 3)]:
        df_num = designs[full].shape[1]-designs[reduced].shape[1]
        df_den = len(y)-designs[full].shape[1]
        sse0, sse1 = fits[reduced][1], fits[full][1]
        statistic = np.maximum((sse0-sse1)/df_num, 0)/(sse1/df_den)
        tests.append(f_dist.sf(statistic, df_num, df_den))
    return dict(overall=overall, direct=direct, transmitted=transmitted,
                p_overall=tests[0], p_direct=tests[1])


def coefficient_screen(predictors, covariates, outcomes, pred_names, outcome_names):
    d = design(covariates, predictors); offset = 1+covariates.shape[1]
    records = []
    for i, outcome in enumerate(outcome_names):
        fitted = sm.OLS(outcomes[:, i], d).fit()
        for j, name in enumerate(pred_names):
            records.append(dict(predictor=name, outcome=outcome,
                beta=float(fitted.params[offset+j]), p=float(fitted.pvalues[offset+j])))
    table = pd.DataFrame(records); table["q"] = bh(table.p.to_numpy())
    return table


def vif_values(values):
    values = np.asarray(values, float)
    output = []
    for i in range(values.shape[1]):
        target = values[:, i]; others = np.delete(values, i, axis=1)
        if np.var(target) <= 0: output.append(np.inf); continue
        d = np.column_stack([np.ones(len(values)), others])
        r2 = float(r2_fit(target[:, None], d)[0][0])
        output.append(np.inf if 1-r2 <= 1e-12 else 1/(1-r2))
    return np.asarray(output)


def remove_pathomic_collinearity(x, m, c, names, threshold=VIF_THRESHOLD):
    names = list(names); indices = list(range(len(names)))
    while indices:
        fixed = np.column_stack([c, x])
        values = vif_values(np.column_stack([fixed, m[:, indices]]))
        if np.max(values) < threshold:
            return indices
        scores = values[fixed.shape[1]:]
        indices.pop(int(np.argmax(scores)))
    raise ValueError("error")


def classify(q_overall, lower, q_direct):
    significant = np.asarray(q_overall) < Q_THRESHOLD
    transmitted = np.asarray(lower) > 0
    direct = np.asarray(q_direct) < Q_THRESHOLD
    labels = np.full(len(significant), "Trans-Null", dtype=object)
    labels[significant & transmitted & ~direct] = "Comp-Trans"
    labels[significant & transmitted & direct] = "Part-Trans"
    labels[significant & ~transmitted & direct] = "Direct-Dom"
    return labels


def infer(x, m, c, y, y_names, n_boot=N_BOOTSTRAP, seed=SEED):
    point = variance_components(x, m, c, y)
    rng = np.random.default_rng(seed); boot = np.empty((n_boot, y.shape[1]))
    for b in range(n_boot):
        index = rng.integers(0, len(y), len(y))
        try:
            boot[b] = variance_components(x[index], m[index], c[index], y[index])["transmitted"]
        except ValueError as exc:
            raise ValueError("error") from exc
    lo, hi = np.quantile(boot, [.025, .975], axis=0)
    result = pd.DataFrame(dict(feature=y_names, **point))
    result["q_overall"] = bh(result.p_overall.to_numpy())
    result["q_direct"] = bh(result.p_direct.to_numpy())
    result["transmitted_lower"], result["transmitted_upper"] = lo, hi
    result["transmission_score"] = np.divide(result.transmitted, result.overall,
        out=np.full(len(result), np.nan), where=result.overall.to_numpy() > 0)
    result["type"] = classify(result.q_overall, lo, result.q_direct)
    return result


def main():
    clinical, xf, mf, yf = [frame(p) for p in [CLINICAL, X_FILE, M_FILE, Y_FILE]]
    ids = xf.index
    if not all(ids.isin(d.index).all() for d in [clinical, mf, yf]):
        raise ValueError("error")
    if not clinical.loc[ids, "cohort"].eq("training").all():
        raise ValueError("error")
    xnames, mnames, ynames = list(xf), list(mf), list(yf)
    x, m, c, y = [np.asarray(a, float) for a in
        [xf.loc[ids], mf.loc[ids], clinical.loc[ids, CLINICAL_COLS], yf.loc[ids]]]
    require_finite(x, m, c, y)
    xm = coefficient_screen(x, c, m, xnames, mnames)
    selected = sorted(set(xm.loc[xm.q < Q_THRESHOLD, "outcome"]))
    if not selected: raise ValueError("error")
    selected_indices = [mnames.index(name) for name in selected]
    subset = m[:, selected_indices]
    retained = remove_pathomic_collinearity(x, subset, c, selected)
    mnames = [selected[i] for i in retained]; m = subset[:, retained]
    my = coefficient_screen(m, c, y, mnames, ynames)
    ynames = sorted(set(my.loc[my.q < Q_THRESHOLD, "outcome"]))
    if not ynames: raise ValueError("error")
    y = yf.loc[ids, ynames].to_numpy(float)
    result = infer(x, m, c, y, ynames)
    chains = []
    for ym in my[(my.q < Q_THRESHOLD) & my.outcome.isin(ynames)].itertuples():
        for xx in xm[(xm.q < Q_THRESHOLD) & (xm.outcome == ym.predictor)].itertuples():
            chains.append(dict(feature=ym.outcome, pathway=xx.predictor,
                pathomic=ym.predictor, beta_xm=xx.beta, beta_my=ym.beta,
                q_xm=xx.q, q_my=ym.q, strength=xx.beta*ym.beta))
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "transmission_features.csv", index=False)
    xm.to_csv(OUT / "pathway_pathomic_coefficients.csv", index=False)
    my.to_csv(OUT / "pathomic_radiomic_coefficients.csv", index=False)
    pd.DataFrame(chains, columns=["feature", "pathway", "pathomic", "beta_xm", "beta_my",
                                "q_xm", "q_my", "strength"]).to_csv(OUT / "supported_chains.csv", index=False)
    (OUT / "frozen_specification.json").write_text(json.dumps(dict(
        xcore=xnames, mcore=mnames, ccore=CLINICAL_COLS, ycore=ynames,
        bootstrap=N_BOOTSTRAP, seed=SEED, q_threshold=Q_THRESHOLD,
        vif_threshold=VIF_THRESHOLD, sample_count=len(ids),
        components="incremental_unadjusted_R2", bootstrap_rank_failure="stop"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

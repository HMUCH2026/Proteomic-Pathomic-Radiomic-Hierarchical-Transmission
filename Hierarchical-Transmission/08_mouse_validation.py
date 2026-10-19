
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "input/mouse/measurements.csv"
MAPPING = ROOT / "input/mouse/correspondence.csv"
OUT = ROOT / "results/mouse"
Q = .05
GATE_ALPHA = .05
GATE_TEST = "two_sided"  
TYPE_ORDER = ["Comp-Trans", "Part-Trans", "Direct-Dom"]
TREND_SCORES = np.array([2., 1., 0.])


def bh(p):
    p = np.asarray(p, float); order = np.argsort(p); q = np.empty(len(p))
    q[order] = np.minimum(1, np.minimum.accumulate(
        (p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]); return q


def allocations(n=12, treated=6):
    return np.array([[int(i in selected) for i in range(n)]
                     for selected in combinations(range(n), treated)], dtype=int)


def exact_p_matrix(values, labels):
    
    a = allocations(len(labels), int(np.sum(labels)))
    contrasts = a/a.sum(1)[:, None]-(1-a)/(1-a).sum(1)[:, None]
    differences = contrasts @ np.asarray(values, float)
    p = np.empty_like(differences)
    for j in range(differences.shape[1]):
        absolute = np.abs(differences[:, j])
        p[:, j] = np.mean(absolute[None, :] >= absolute[:, None]-1e-12, axis=1)
    observed = np.flatnonzero(np.all(a == labels, axis=1))
    if len(observed) != 1: raise ValueError("error")
    return a, differences, p, int(observed[0])


def trend_statistic(n, total, scores=TREND_SCORES):
    n, total = np.asarray(n, float), np.asarray(total, float)
    if total.sum() <= 1: return np.nan
    p = n.sum()/total.sum(); mean_score = np.sum(scores*total)/total.sum()
    numerator = np.sum(scores*(n-total*p))
    variance = p*(1-p)*np.sum(total*(scores-mean_score)**2)
    return float(numerator/np.sqrt(variance)) if variance > 0 else 0.


def main():
    d = pd.read_csv(DATA, dtype={"mouse_id": str}); mapping = pd.read_csv(MAPPING)
    if len(d) != 12 or d.mouse_id.duplicated().any(): raise ValueError("error")
    labels = d.treatment.to_numpy(int)
    if set(labels) != {0, 1} or labels.sum() != 6: raise ValueError("error")
    if not set(mapping.type) <= set(TYPE_ORDER): raise ValueError("error")
    if not mapping.qc_pass.isin([0, 1]).all(): raise ValueError("error")
    pairs = mapping[["human_feature", "mouse_feature"]].drop_duplicates()
    if pairs.human_feature.duplicated().any() or pairs.mouse_feature.duplicated().any():
        raise ValueError("error")
    if mapping.groupby("mouse_feature").qc_pass.nunique().max() != 1:
        raise ValueError("error")
    raw = mapping.copy(); mapping = mapping[mapping.qc_pass == 1].copy()
    if mapping.empty: raise ValueError("error")
    for feature, g in mapping.groupby("mouse_feature"):
        if g.type.nunique() != 1: raise ValueError("error")
    mri = list(dict.fromkeys(mapping.mouse_feature)); pathology = list(dict.fromkeys(mapping.pathomic_feature))
    names = mri+pathology
    if len(set(names)) != len(names): raise ValueError("error")
    if not set(names+["pRB_RB", "Ki67"]) <= set(d): raise ValueError("error")
    values = d[names+["pRB_RB", "Ki67"]].to_numpy(float)
    if not np.isfinite(values).all(): raise ValueError("error")
    a, diffs, pvals, observed = exact_p_matrix(values, labels)
    gate_p, gate_diff = pvals[observed, -2:], diffs[observed, -2:]
    gate_pass = bool(np.all(gate_p < GATE_ALPHA) and np.all(gate_diff < 0))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pharmacodynamic_gate.json").write_text(json.dumps(dict(pRB_RB_p=float(gate_p[0]),
        Ki67_p=float(gate_p[1]), differences=gate_diff.tolist(), passed=gate_pass,
        criterion="two-sided exact p<alpha AND drug mean lower, both endpoints"), indent=2), encoding="utf-8")
    if not gate_pass:
        raise RuntimeError("error")
    qmri = np.vstack([bh(row[:len(mri)]) for row in pvals])
    qpath = np.vstack([bh(row[len(mri):len(names)]) for row in pvals])
    pathology_links = {f: [pathology.index(p) for p in sorted(set(g.pathomic_feature))]
                      for f, g in mapping.groupby("mouse_feature")}
    type_map = mapping.drop_duplicates("mouse_feature").set_index("mouse_feature").type
    N = np.array([sum(type_map[f] == t for f in mri) for t in TYPE_ORDER])
    if np.any(N == 0): raise ValueError("error")
    responsive, concordant, statistics = [], [], []
    for b in range(len(a)):
        response = qmri[b] < Q
        concord = np.array([response[j] and np.any(qpath[b, pathology_links[f]] < Q)
                            for j, f in enumerate(mri)])
        counts_m = np.array([sum(response[j] for j, f in enumerate(mri) if type_map[f] == t)
                             for t in TYPE_ORDER])
        counts_n = np.array([sum(concord[j] for j, f in enumerate(mri) if type_map[f] == t)
                             for t in TYPE_ORDER])
        responsive.append(counts_m); concordant.append(counts_n)
        statistics.append(trend_statistic(counts_n, N))
    statistics = np.asarray(statistics)
    p_trend = float(np.mean(statistics >= statistics[observed]-1e-12))
    rows = []
    for j, t in enumerate(TYPE_ORDER):
        n0 = raw[raw.type == t].mouse_feature.nunique()
        mm, nn = int(responsive[observed][j]), int(concordant[observed][j])
        rows.append(dict(type=t, N0=n0, QC_excluded=n0-int(N[j]), N=int(N[j]), m=mm, n=nn,
                         m_over_N=mm/N[j], n_over_m=nn/mm if mm else np.nan, n_over_N=nn/N[j]))
    pd.DataFrame(rows).to_csv(OUT / "table2_counts.csv", index=False)
    pd.DataFrame(dict(feature=names, difference=diffs[observed, :len(names)],
        p=pvals[observed, :len(names)], q=np.r_[qmri[observed], qpath[observed]],
        domain=["MRI"]*len(mri)+["pathomic"]*len(pathology))).to_csv(OUT / "feature_responses.csv", index=False)
    pd.DataFrame(dict(allocation=np.arange(len(a)), trend=statistics,
        observed=np.arange(len(a)) == observed)).to_csv(OUT / "all_allocations.csv", index=False)
    (OUT / "trend_result.json").write_text(json.dumps(dict(statistic=float(statistics[observed]),
        one_sided_exact_p=p_trend, n_allocations=len(a), scores=TREND_SCORES.tolist(),
        feature_q=Q), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

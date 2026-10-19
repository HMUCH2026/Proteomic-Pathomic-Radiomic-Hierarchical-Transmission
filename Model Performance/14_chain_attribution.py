
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/chain_attribution"
N_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000
SEED = 00
Q = .05


def attribute_chains(chains, importance):
    records = chains.merge(importance[["feature", "mean_abs_shap"]], on="feature", how="inner",
                          validate="many_to_one")
    records = records[(records.q_xm < Q) & (records.q_my < Q)].copy()
    if records.duplicated(["feature", "pathway", "pathomic"]).any():
        raise ValueError("error")
    if set(importance.feature)-set(records.feature):
        raise ValueError("error")
    records["signed_strength"] = records.beta_xm*records.beta_my
    records["absolute_strength"] = records.signed_strength.abs()
    denominator = records.groupby("feature").absolute_strength.transform("sum")
    if (denominator <= 0).any(): raise ValueError("error")
    records["weight"] = records.absolute_strength/denominator
    records["allocated_shap"] = records.weight*records.mean_abs_shap
    records["dominant"] = False
    chosen = records.sort_values(["feature", "absolute_strength", "pathway", "pathomic"],
        ascending=[True, False, True, True]).drop_duplicates("feature").index
    records.loc[chosen, "dominant"] = True
    allocated = records.groupby("feature").allocated_shap.sum()
    expected = importance.set_index("feature").mean_abs_shap.loc[allocated.index]
    if not np.allclose(allocated, expected): raise AssertionError("error")
    pathways = records.groupby("pathway").agg(allocated_shap=("allocated_shap", "sum"),
        n_supported_chains=("feature", "size"), n_features=("feature", "nunique"),
        n_dominant_chains=("dominant", "sum")).reset_index()
    pathways["importance_share"] = pathways.allocated_shap/importance.mean_abs_shap.sum()
    if not np.isclose(pathways.importance_share.sum(), 1): raise AssertionError("error")
    return records, pathways


def rank_concordance(scores, importance, permutations=N_PERMUTATIONS, bootstrap=N_BOOTSTRAP, seed=SEED):
    a, b = np.asarray(scores, float), np.asarray(importance, float)
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("error")
    estimate = float(spearmanr(a, b).statistic)
    if not np.isfinite(estimate): raise ValueError("error")
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(a, rng.permutation(b)).statistic for _ in range(permutations)])
    p = float((1+np.sum(np.abs(null) >= abs(estimate)-1e-12))/(permutations+1))
    draws = []
    for _ in range(bootstrap):
        index = rng.integers(0, len(a), len(a))
        draws.append(spearmanr(a[index], b[index]).statistic)
    values = np.asarray(draws); invalid = int(np.sum(~np.isfinite(values)))
    lo, hi = np.quantile(values, [.025, .975]) if not invalid else (np.nan, np.nan)
    return dict(spearman=estimate, permutation_p=p, lower=float(lo), upper=float(hi),
                undefined_bootstraps=invalid), null, values


def main():
    transmission = pd.read_csv(ROOT / "results/transmission/transmission_features.csv")
    importance = pd.read_csv(ROOT / "results/shap/importance.csv")
    chains = pd.read_csv(ROOT / "results/transmission/supported_chains.csv")
    data = importance.merge(transmission[["feature", "transmission_score", "type"]],
                            on="feature", validate="one_to_one")
    if not data.type.eq("Comp-Trans").all(): raise ValueError("error")
    data["transmission_rank"] = data.transmission_score.rank(method="average", ascending=False)
    data["importance_rank"] = data.mean_abs_shap.rank(method="average", ascending=False)
    summary, null, boot = rank_concordance(data.transmission_score, data.mean_abs_shap)
    allocated, pathways = attribute_chains(chains, importance)
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "feature_concordance.csv", index=False)
    allocated.to_csv(OUT / "chain_allocations.csv", index=False)
    pathways.to_csv(OUT / "pathway_shares.csv", index=False)
    pd.DataFrame(dict(permuted_rho=null)).to_csv(OUT / "permutation_statistics.csv", index=False)
    pd.DataFrame(dict(bootstrap_rho=boot)).to_csv(OUT / "bootstrap_statistics.csv", index=False)
    (OUT / "concordance.json").write_text(json.dumps(dict(**summary,
        permutations=N_PERMUTATIONS, bootstrap=N_BOOTSTRAP, seed=SEED,
        resampling_unit="paired_feature", interpretation="descriptive_association_allocation"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

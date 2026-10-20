
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/transmission_external"
N_BOOTSTRAP = 2000
SEED = 00
KAPPA_FEATURE_BOOTSTRAP = False  
ASSESSABILITY_FILE = ROOT / "input/external_transmission_assessability.csv"
FEATURE_METADATA_FILE = (
    ROOT.parent / "Radiomic/results/features/feature_metadata.csv"
)
VALIDATION_STATUSES = ["agreement", "migration", "not_assessable"]


def annotate_comparison(comparison, metadata):
    """Attach extraction metadata and distinguish the three validation outcomes."""
    required = ["feature", "source", "family", "compartment"]
    if not set(required) <= set(metadata.columns):
        raise ValueError("Feature metadata must contain: " + ", ".join(required))
    if (metadata[required].isna().any().any() or
            metadata.feature.duplicated().any()):
        raise ValueError("Feature metadata contains missing values or duplicate features")
    result = comparison.merge(
        metadata[required], on="feature", how="left", validate="one_to_one"
    )
    if result[required].isna().any().any():
        missing = result.loc[result[required].isna().any(axis=1), "feature"]
        raise ValueError("Missing metadata for: " + ", ".join(missing.astype(str)))
    if (result.assessable.isna().any() or
            not result.assessable.isin([True, False]).all()):
        raise ValueError("Assessability must be a non-missing Boolean indicator")
    assessed = result.assessable.astype(bool)
    if result.type_train.isna().any() or result.loc[assessed, "type_external"].isna().any():
        raise ValueError("Training or assessable external type assignments are missing")
    result["validation_status"] = np.select(
        [~assessed, result.type_train == result.type_external],
        ["not_assessable", "agreement"], default="migration"
    )
    return result


def stratified_agreement(comparison, grouping):
    """Count all training features, using only assessable ones for agreement rates.

    *_fraction uses n_training as denominator and sums to one across the three
    validation states. exact_agreement_rate uses n_assessable as denominator;
    it is undefined (NaN) when a stratum has no assessable features.
    """
    if comparison[grouping].isna().any():
        raise ValueError("Missing grouping labels: " + grouping)
    if not comparison.validation_status.isin(VALIDATION_STATUSES).all():
        raise ValueError("Unknown validation status")
    counts = (
        comparison.groupby([grouping, "validation_status"], observed=True)
        .size().unstack(fill_value=0)
        .reindex(columns=VALIDATION_STATUSES, fill_value=0)
    )
    counts.columns.name = None
    counts["n_training"] = counts[VALIDATION_STATUSES].sum(axis=1)
    counts["n_assessable"] = counts.agreement + counts.migration
    counts["exact_agreement_rate"] = (
        counts.agreement / counts.n_assessable.replace(0, np.nan)
    )
    for status in VALIDATION_STATUSES:
        counts[status + "_fraction"] = counts[status] / counts.n_training
    fractions = [status + "_fraction" for status in VALIDATION_STATUSES]
    if (counts.n_training.sum() != len(comparison) or
            not np.allclose(counts[fractions].sum(axis=1), 1.)):
        raise AssertionError("Stratified counts or fractions are inconsistent")
    return counts.reset_index()


def load_scientific_module():
    spec = importlib.util.spec_from_file_location("hierarchical_transmission",
        ROOT / "06_hierarchical_transmission.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    # Load metadata before the computationally intensive bootstrap analysis.
    if not FEATURE_METADATA_FILE.exists():
        raise FileNotFoundError("Feature metadata not found: " + str(FEATURE_METADATA_FILE))
    metadata = pd.read_csv(FEATURE_METADATA_FILE)
    core = load_scientific_module()
    spec = json.loads((ROOT / "results/transmission/frozen_specification.json").read_text())
    trained = pd.read_csv(ROOT / "results/transmission/transmission_features.csv")
    x, m, y, clinical = [core.frame(p) for p in [
        ROOT / "results/pathways/external1_scores.csv",
        ROOT / "results/qc/pathomics_scaled.csv",
        ROOT / "results/qc/radiomics_scaled.csv", ROOT / "input/clinical.csv"]]
    ids = x.index
    if not all(ids.isin(d.index).all() for d in [m, y, clinical]):
        raise ValueError("error")
    if not clinical.loc[ids, "cohort"].eq("external1").all():
        raise ValueError("error")
    for cols, data in [(spec["xcore"], x), (spec["mcore"], m), (spec["ccore"], clinical)]:
        if not set(cols) <= set(data): raise ValueError("error")
    if not ASSESSABILITY_FILE.exists():
        raise ValueError("error")
    assessability = pd.read_csv(ASSESSABILITY_FILE)
    if (assessability.feature.duplicated().any() or
            not set(spec["ycore"]) <= set(assessability.feature) or
            not assessability.assessable.isin([0, 1]).all()):
        raise ValueError("error")
    assessed = assessability.loc[(assessability.assessable == 1) &
                                 assessability.feature.isin(spec["ycore"]), "feature"].tolist()
    if any(name not in y or not np.isfinite(y.loc[ids, name].to_numpy(float)).all()
           for name in assessed):
        raise ValueError("error")
    if not assessed: raise ValueError("error")
    external = core.infer(x.loc[ids, spec["xcore"]].to_numpy(float),
        m.loc[ids, spec["mcore"]].to_numpy(float),
        clinical.loc[ids, spec["ccore"]].to_numpy(float),
        y.loc[ids, assessed].to_numpy(float), assessed, N_BOOTSTRAP, SEED)
    comparison = trained.merge(external, on="feature", how="left", suffixes=("_train", "_external"))
    comparison["assessable"] = comparison.feature.isin(assessed)
    if "reason" in assessability:
        comparison = comparison.merge(assessability[["feature", "reason"]], on="feature", how="left")
    comparison["exact_agreement"] = np.where(comparison.assessable,
        comparison.type_train == comparison.type_external, np.nan)
    comparison = annotate_comparison(comparison, metadata)
    stratified = {
        grouping: stratified_agreement(comparison, grouping)
        for grouping in ["source", "family"]
    }
    paired = comparison[comparison.assessable]
    kapp = cohen_kappa_score(paired.type_train, paired.type_external)
    summary = dict(n_training=len(trained), n_assessable=len(paired),
        n_not_assessable=len(trained)-len(paired), exact_agreement=float(
            (paired.type_train == paired.type_external).mean()), kappa=float(kapp))
    for name in ["transmitted", "direct"]:
        a, b = paired[f"{name}_train"], paired[f"{name}_external"]
        if np.std(a) and np.std(b):
            summary[name+"_pearson"] = float(pearsonr(a, b).statistic)
            summary[name+"_spearman"] = float(spearmanr(a, b).statistic)
    if KAPPA_FEATURE_BOOTSTRAP:
        rng = np.random.default_rng(SEED); vals = []
        for _ in range(N_BOOTSTRAP):
            z = paired.iloc[rng.integers(0, len(paired), len(paired))]
            vals.append(cohen_kappa_score(z.type_train, z.type_external))
        if not np.isfinite(vals).all(): raise ValueError("error")
        summary["kappa_feature_bootstrap_CI"] = np.quantile(vals, [.025, .975]).tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT / "feature_type_agreement.csv", index=False)
    for grouping, table in stratified.items():
        table.to_csv(OUT / f"agreement_by_{grouping}.csv", index=False)
    pd.crosstab(paired.type_train, paired.type_external).to_csv(OUT / "migration_matrix.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

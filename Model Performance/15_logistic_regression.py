
from pathlib import Path
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, ConvergenceWarning

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/logistic"
MODEL_STRATEGY = None  
PROBABILITY_UNIT = 1.0
ENTRY_P = .10
COHORTS = ["training", "external1", "external2", "public"]
SCORES = ["Comp-Trans", "Part-Trans", "Direct-Dom", "Trans-Null", "Data-Driven"]
BIOMARKERS = ["ER", "PR", "HR", "HER2", "Ki67"]  

COVARIATES = ["age", "menopausal", "tumor_diameter_mm", "histology_nonductal",
              "grade_III", "clinical_T_high", "clinical_N_positive"]
PUBLIC_COVARIATES = ["age", "menopausal", "tumor_diameter_mm", "grade_III"]
PUBLIC_BIOMARKERS = ["HR", "HER2"]
MAX_ITER = 200


def fit_wald(x, y):
    names = list(x); d = sm.add_constant(x, has_constant="add")
    if len(y) <= d.shape[1] or np.linalg.matrix_rank(d.to_numpy(float)) != d.shape[1]:
        raise ValueError("error")
    if not np.isfinite(d.to_numpy(float)).all() or not np.isfinite(y).all():
        raise ValueError("error")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.simplefilter("error", RuntimeWarning)
            model = sm.Logit(np.asarray(y, int), d).fit(disp=False, maxiter=MAX_ITER)
    except (np.linalg.LinAlgError, PerfectSeparationWarning, ConvergenceWarning, RuntimeWarning) as exc:
        raise ValueError("error") from exc
    if not model.mle_retvals.get("converged", False): raise ValueError("error")
    rows = []
    critical = norm.ppf(.975)
    for name in names:
        beta, se = float(model.params[name]), float(model.bse[name])
        if not np.isfinite([beta, se]).all() or se <= 0: raise ValueError("error")
        lower_log, upper_log = beta-critical*se, beta+critical*se
        if max(abs(lower_log), abs(upper_log), abs(beta)) > 700:
            raise ValueError("error")
        rows.append(dict(variable=name, beta=beta, standard_error=se, OR=float(np.exp(beta)),
            lower=float(np.exp(lower_log)), upper=float(np.exp(upper_log)),
            p=float(2*norm.sf(abs(beta/se))), n=len(y), events=int(np.sum(y))))
    return rows


def main():
    if MODEL_STRATEGY not in {"joint", "per_score"}:
        raise ValueError("AUTHOR_INPUT_NEEDED: error")
    clinical = pd.read_csv(ROOT / "input/clinical.csv", dtype={"case_id": str}).set_index("case_id")
    predictions = pd.read_csv(ROOT / "results/models/predictions.csv", dtype={"case_id": str}).set_index("case_id")
    data = clinical.join(predictions[SCORES], how="left", validate="one_to_one")
    data[SCORES] = data[SCORES]/PROBABILITY_UNIT
    univ, multiv, cache = [], [], {}
    for cohort in COHORTS:
        group = data[data.cohort == cohort].sort_index()
        covariates = PUBLIC_COVARIATES if cohort == "public" else COVARIATES
        biomarkers = PUBLIC_BIOMARKERS if cohort == "public" else BIOMARKERS
        candidates = biomarkers+SCORES
        required = covariates+candidates
        if not set(required) <= set(group) or group[required+["pCR"]].isna().any().any():
            raise ValueError(f"{cohort}: error")
        y = group.pCR.to_numpy(int)
        if set(y) != {0, 1}: raise ValueError("error")
        pmap = {}
        for name in required:
            content = json.dumps(group.index.tolist()).encode()+group[[name, "pCR"]].to_numpy(float).tobytes()
            signature = hashlib.sha256(content).hexdigest()
            if signature not in cache: cache[signature] = fit_wald(group[[name]], y)[0]
            result = dict(cache[signature]); result["variable"] = name
            pmap[name] = result["p"]
            univ.append(dict(cohort=cohort, analysis="univariable", **result))
        selected = [v for v in candidates if pmap[v] < ENTRY_P]
        if MODEL_STRATEGY == "joint":
            models = {"joint": covariates+selected}
        else:
            selected_biomarkers = [v for v in biomarkers if v in selected]
            models = {score: covariates+selected_biomarkers+([score] if score in selected else [])
                      for score in SCORES}
        for model_name, columns in models.items():
            for result in fit_wald(group[columns], y):
                multiv.append(dict(cohort=cohort, model=model_name, analysis="multivariable", **result))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(univ).to_csv(OUT / "univariable.csv", index=False)
    pd.DataFrame(multiv).to_csv(OUT / "multivariable.csv", index=False)
    (OUT / "parameters.json").write_text(json.dumps(dict(strategy=MODEL_STRATEGY,
        score_probability_unit=PROBABILITY_UNIT, candidate_entry_p=ENTRY_P,
        covariates_forced=True, interval="exp(beta +/- qnorm(.975)*SE)",
        p_method="two-sided Wald", missing_clinical="error"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

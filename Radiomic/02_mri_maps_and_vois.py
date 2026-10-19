from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "results/mri/preprocessed_manifest.csv"
BREAST_MASKS = ROOT / "input/breast_masks.csv"
OUT = ROOT / "results/vois"
RING_MM = 5.
PERCENTILES = (1., 99.)
CLIP_SD = 3.
EPS = 1e-12


def read_image(path):
    p = Path(str(path))
    return sitk.ReadImage(str(p if p.is_absolute() else ROOT / p))


def same_geometry(a, b):
    return (a.GetSize() == b.GetSize() and
            np.allclose(a.GetSpacing(), b.GetSpacing()) and
            np.allclose(a.GetOrigin(), b.GetOrigin()) and
            np.allclose(a.GetDirection(), b.GetDirection()))


def parametric_maps(dw, b_values, dce, times):
    maps = {}
    if dw is not None:
        b = np.asarray(b_values, float)
        if len(np.unique(b)) < 2 or not np.isclose(b.min(), 0):
            raise ValueError("error")
        valid = np.all(np.asarray(dw) > 0, axis=0)
        log_s = np.log(np.maximum(dw, EPS))
        centered = b - b.mean()
        adc = -np.tensordot(centered, log_s, axes=(0, 0)) / np.sum(centered**2)
        maps["ADC"] = np.where(valid, adc, np.nan)
    if dce is not None:
        times = np.asarray(times, float)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("error")
        dce = np.asarray(dce, float)
        peak = np.argmax(dce, axis=0)
        maximum = np.max(dce, axis=0)
        baseline = dce[0]
        valid = baseline > EPS
        mu = np.divide(maximum-baseline, baseline,
                       out=np.full_like(baseline, np.nan), where=valid)
        tp = times[peak] - times[0]
        ur = np.divide(mu, tp, out=np.zeros_like(mu), where=tp > 0)
        remaining = times[-1] - times[peak]
        wr = np.divide(maximum-dce[-1], baseline*remaining,
                       out=np.zeros_like(mu), where=valid & (remaining > 0))
        for name, values in [("MU", mu), ("TP", tp), ("UR", ur), ("WR", wr)]:
            maps[name] = np.where(valid, values, np.nan)
    return maps


def preprocess_voi(values, mask):
    v = np.asarray(values, float)[mask]
    if not len(v) or not np.isfinite(v).all():
        raise ValueError("error")
    lo, hi = np.percentile(v, PERCENTILES)
    v = np.clip(v, lo, hi)
    mean, sd = float(v.mean()), float(v.std(ddof=0))
    if sd <= EPS:
        raise ValueError("error")
    v = (v-mean)/sd
    mu, sigma = float(v.mean()), float(v.std(ddof=0))
    out = np.zeros_like(values, dtype=np.float64)
    out[mask] = np.clip(v, mu-CLIP_SD*sigma, mu+CLIP_SD*sigma)
    return out, {"p1": float(lo), "p99": float(hi), "mean": mean, "sd": sd}


def main():
    manifest = pd.read_csv(MANIFEST, dtype={"case_id": str, "rater": str})
    breasts = (pd.read_csv(BREAST_MASKS, dtype={"case_id": str, "rater": str})
               if BREAST_MASKS.exists() else pd.DataFrame())
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for (case, species, rater, seq), frame in manifest.groupby(
            ["case_id", "species", "rater", "sequence"], sort=False):
        key = "b_value" if seq == "DW" else "time_seconds"
        frame = frame.sort_values(key)
        images = [read_image(p) for p in frame.image]
        ref = images[0]
        if not all(same_geometry(im, ref) for im in images):
            raise ValueError(f"{case}/{seq}: error")
        mask_im = read_image(frame.iloc[0]["mask"])
        if not same_geometry(ref, mask_im):
            raise ValueError("error")
        tumor = sitk.GetArrayFromImage(mask_im) > 0
        if not tumor.any():
            raise ValueError("error")
        masks = {"IT": tumor}
        if species == "human":
            if breasts.empty:
                raise ValueError("error")
            matched = breasts[(breasts.case_id == case) & (breasts.rater == rater) &
                              (breasts.sequence == seq)]
            if len(matched) != 1:
                raise ValueError("error")
            breast = read_image(matched.iloc[0]["mask"])
            if not same_geometry(ref, breast):
                raise ValueError("error")
            distance = distance_transform_edt(~tumor, sampling=ref.GetSpacing()[::-1])
            masks["PT5mm"] = ((distance <= RING_MM) & ~tumor &
                               (sitk.GetArrayFromImage(breast) > 0))
        arrays = np.stack([sitk.GetArrayFromImage(im) for im in images])
        if seq == "FS-T2W":
            maps = {"FS-T2W": arrays[0]}
        elif seq == "DW":
            maps = parametric_maps(arrays, frame.b_value.to_numpy(), None, None)
        else:
            maps = parametric_maps(None, None, arrays, frame.time_seconds.to_numpy())
        folder = OUT / str(case) / str(rater)
        folder.mkdir(parents=True, exist_ok=True)
        for compartment, mask in masks.items():
            mpath = folder / f"{seq}_{compartment}_mask.nii.gz"
            im = sitk.GetImageFromArray(mask.astype(np.uint8)); im.CopyInformation(ref)
            sitk.WriteImage(im, str(mpath))
            for source, values in maps.items():
                raw_path = folder / f"{source}_raw.nii.gz"
                raw = sitk.GetImageFromArray(np.asarray(values, dtype=float)); raw.CopyInformation(ref)
                sitk.WriteImage(raw, str(raw_path))
                normalized, stats = preprocess_voi(values, mask)
                ipath = folder / f"{source}_{compartment}_normalized.nii.gz"
                im = sitk.GetImageFromArray(normalized); im.CopyInformation(ref)
                sitk.WriteImage(im, str(ipath))
                records.append(dict(case_id=case, species=species, rater=rater,
                                    source=source, compartment=compartment,
                                    image=str(ipath), mask=str(mpath), **stats))
    pd.DataFrame(records).to_csv(OUT / "voi_manifest.csv", index=False)
    (OUT / "parameters.json").write_text(json.dumps({"ring_mm": RING_MM,
        "percentiles": PERCENTILES, "clip_sd": CLIP_SD,
        "peak_last_WR": 0, "tp_zero_UR": 0}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

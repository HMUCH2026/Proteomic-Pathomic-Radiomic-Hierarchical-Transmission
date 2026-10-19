from pathlib import Path
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "input/mri_manifest.csv"
OUT = ROOT / "results/mri"
SPACING = {"human": (1., 1., 1.), "mouse": (.125, .125, 1.)}
N4_ITERATIONS = [50, 50, 30, 20]
REGISTRATION_SEED = 00
REGISTRATION_THREADS = 1
N4_MOUSE = False  


def path_value(value):
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def nonempty(value):
    return pd.notna(value) and str(value).strip() != ""


def n4_field(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    foreground = sitk.OtsuThreshold(image, 0, 1, 200)
    filt = sitk.N4BiasFieldCorrectionImageFilter()
    filt.SetMaximumNumberOfIterations(N4_ITERATIONS)
    filt.Execute(image, foreground)
    return sitk.Exp(filt.GetLogBiasFieldAsImage(image))


def register(moving, reference):
    initial = sitk.CenteredTransformInitializer(
        reference, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(.2, REGISTRATION_SEED)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(2., .001, 200)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2., 1., 0.])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    return reg.Execute(reference, moving)


def target_grid(reference, spacing):
    size = [int(round((n - 1) * s / t)) + 1
            for n, s, t in zip(reference.GetSize(), reference.GetSpacing(), spacing)]
    grid = sitk.Image(size, sitk.sitkFloat32)
    grid.SetSpacing(spacing)
    grid.SetOrigin(reference.GetOrigin())
    grid.SetDirection(reference.GetDirection())
    return grid


def resample(image, reference, transform=None, mask=False):
    transform = transform or sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(image, reference, transform,
                         sitk.sitkNearestNeighbor if mask else sitk.sitkBSpline,
                         0., sitk.sitkUInt8 if mask else sitk.sitkFloat32)


def main():
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(REGISTRATION_THREADS)
    table = pd.read_csv(MANIFEST, dtype={"case_id": str, "rater": str})
    required = {"case_id", "species", "rater", "sequence", "image", "mask",
                "b_value", "time_seconds"}
    if not required <= set(table):
        raise ValueError("error")
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for (case, species, rater, seq), frame in table.groupby(
            ["case_id", "species", "rater", "sequence"], sort=False):
        if species not in SPACING or seq not in {"FS-T2W", "DW", "DCE"}:
            raise ValueError("error")
        key = "b_value" if seq == "DW" else "time_seconds"
        if seq != "FS-T2W" and frame[key].isna().any():
            raise ValueError("error")
        frame = frame.sort_values(key, na_position="first").reset_index(drop=True)
        if seq == "FS-T2W" and len(frame) != 1:
            raise ValueError("error")
        images = [sitk.ReadImage(str(path_value(p)), sitk.sitkFloat32)
                  for p in frame.image]
        if any(im.GetDimension() != 3 for im in images):
            raise ValueError("error")
        do_n4 = species == "human" or N4_MOUSE
        if do_n4 and seq != "DCE":
            images = [im / n4_field(im) for im in images]
        reference = images[0]
        transforms = [sitk.Transform(3, sitk.sitkIdentity)]
        transforms += [register(im, reference) for im in images[1:]]
        registered = [resample(im, reference, tr) for im, tr in zip(images, transforms)]
        if do_n4 and seq == "DCE":
            
            field = n4_field(registered[0])
            registered = [im / field for im in registered]
        available = [i for i, value in enumerate(frame["mask"]) if nonempty(value)]
        if not available:
            raise ValueError("error")
        masks = []
        for i in available:
            mask = sitk.ReadImage(str(path_value(frame.loc[i, "mask"]))) > 0
            if (mask.GetSize() != images[i].GetSize() or
                    not np.allclose(mask.GetSpacing(), images[i].GetSpacing()) or
                    not np.allclose(mask.GetOrigin(), images[i].GetOrigin()) or
                    not np.allclose(mask.GetDirection(), images[i].GetDirection())):
                raise ValueError("error")
            masks.append(resample(mask, reference, transforms[i], True))
        if any(not np.array_equal(sitk.GetArrayFromImage(masks[0]),
                                  sitk.GetArrayFromImage(m)) for m in masks[1:]):
            raise ValueError("error")
        grid = target_grid(reference, SPACING[species])
        folder = OUT / str(case) / str(rater) / str(seq)
        folder.mkdir(parents=True, exist_ok=True)
        mask_path = folder / "intratumoral.nii.gz"
        sitk.WriteImage(resample(masks[0], grid, mask=True), str(mask_path))
        for i, (row, im) in enumerate(zip(frame.to_dict("records"), registered)):
            image_path = folder / f"volume_{i:03d}.nii.gz"
            sitk.WriteImage(resample(im, grid), str(image_path))
            row.update(image=str(image_path), mask=str(mask_path))
            records.append(row)
    pd.DataFrame(records).to_csv(OUT / "preprocessed_manifest.csv", index=False)
    (OUT / "parameters.json").write_text(json.dumps({
        "spacing": SPACING, "n4_iterations": N4_ITERATIONS,
        "n4_mouse": N4_MOUSE, "registration_seed": REGISTRATION_SEED,
        "SimpleITK": sitk.Version_VersionString()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

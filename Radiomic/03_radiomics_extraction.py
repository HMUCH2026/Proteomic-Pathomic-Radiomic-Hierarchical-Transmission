from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "results/vois/voi_manifest.csv"
OUT = ROOT / "results/features"
BIN_COUNT_T2 = 32
BIN_WIDTHS = {"ADC": None, "MU": None, "TP": None, "UR": None, "WR": None}
SHAPE = ["SphericalDisproportion", "Sphericity", "MeshVolume", "VoxelVolume",
         "SurfaceArea", "SurfaceVolumeRatio", "Maximum3DDiameter", "MajorAxisLength",
         "MinorAxisLength", "Elongation", "Maximum2DDiameterColumn", "Maximum2DDiameterRow",
         "Maximum2DDiameterSlice", "LeastAxisLength", "Flatness"]
FIRST = ["10Percentile", "90Percentile", "Energy", "Entropy", "InterquartileRange",
         "Kurtosis", "Maximum", "MeanAbsoluteDeviation", "Mean", "Median", "Minimum",
         "Range", "RobustMeanAbsoluteDeviation", "RootMeanSquared", "Skewness",
         "TotalEnergy", "Uniformity", "Variance"]
GLCM = ["Autocorrelation", "ClusterProminence", "ClusterShade", "ClusterTendency",
        "Contrast", "Correlation", "DifferenceAverage", "DifferenceEntropy",
        "DifferenceVariance", "Id", "Idm", "Idmn", "Idn", "Imc1", "Imc2",
        "InverseVariance", "JointAverage", "JointEnergy", "JointEntropy", "MCC",
        "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares"]
GLRLM = ["GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "GrayLevelVariance",
         "HighGrayLevelRunEmphasis", "LongRunEmphasis", "LongRunHighGrayLevelEmphasis",
         "LongRunLowGrayLevelEmphasis", "LowGrayLevelRunEmphasis", "RunEntropy",
         "RunLengthNonUniformity", "RunLengthNonUniformityNormalized", "RunPercentage",
         "RunVariance", "ShortRunEmphasis", "ShortRunHighGrayLevelEmphasis",
         "ShortRunLowGrayLevelEmphasis"]
GLSZM = ["GrayLevelNonUniformity", "GrayLevelNonUniformityNormalized", "LargeAreaEmphasis",
         "GrayLevelVariance", "HighGrayLevelZoneEmphasis", "ZoneVariance",
         "LargeAreaHighGrayLevelEmphasis", "ZoneEntropy", "LargeAreaLowGrayLevelEmphasis",
         "LowGrayLevelZoneEmphasis", "SizeZoneNonUniformity", "SmallAreaEmphasis",
         "SizeZoneNonUniformityNormalized", "SmallAreaHighGrayLevelEmphasis",
         "SmallAreaLowGrayLevelEmphasis", "ZonePercentage"]
GLDM = ["DependenceEntropy", "DependenceNonUniformity", "DependenceNonUniformityNormalized",
        "DependenceVariance", "GrayLevelNonUniformity", "GrayLevelVariance",
        "HighGrayLevelEmphasis", "LargeDependenceEmphasis", "LargeDependenceHighGrayLevelEmphasis",
        "LargeDependenceLowGrayLevelEmphasis", "SmallDependenceHighGrayLevelEmphasis",
        "SmallDependenceLowGrayLevelEmphasis", "SmallDependenceEmphasis", "LowGrayLevelEmphasis"]
NGTDM = ["Busyness", "Coarseness", "Complexity", "Contrast", "Strength"]
FEATURES = {"shape": SHAPE, "firstorder": FIRST, "glcm": GLCM,
            "glrlm": GLRLM, "glszm": GLSZM, "gldm": GLDM, "ngtdm": NGTDM}


def extract_one(image, mask, source):
    import SimpleITK as sitk
    from radiomics import featureextractor
    settings = dict(normalize=False, resampledPixelSpacing=None, label=1,
                    correctMask=False, additionalInfo=False, distances=[1])
    if source == "FS-T2W":
        settings["binCount"] = BIN_COUNT_T2
    else:
        width = BIN_WIDTHS.get(source)
        if width is None or width <= 0:
            raise ValueError(f"error")
        settings["binWidth"] = width
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes(); extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    
    extractor.enableFeaturesByName(**FEATURES)
    image_im, mask_im = sitk.ReadImage(str(image)), sitk.ReadImage(str(mask))
    values = extractor.execute(image_im, mask_im)
    selected = {f"{family}_{name}": float(np.asarray(values[f"original_{family}_{name}"]))
                for family, names in FEATURES.items() for name in names}
    if len(selected) != 108 or not np.isfinite(list(selected.values())).all():
        raise ValueError("error")
    return selected


def main():
    import radiomics
    table = pd.read_csv(MANIFEST, dtype={"case_id": str, "rater": str})
    if any(BIN_WIDTHS.get(s) is None for s in set(table.source)-{"FS-T2W"}):
        raise ValueError("error")
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for rater, frame in table.groupby("rater"):
        rows = []
        for case, cases in frame.groupby("case_id", sort=False):
            result = {"case_id": case}
            expected = 12 if cases.iloc[0].species == "human" else 6
            if len(cases) != expected or cases.duplicated(["source", "compartment"]).any():
                raise ValueError("error")
            for row in cases.itertuples():
                values = extract_one(row.image, row.mask, row.source)
                for name, value in values.items():
                    feature = f"{row.compartment}_{row.source}_{name}"
                    result[feature] = value
                    metadata[feature] = dict(feature=feature, source=row.source,
                        compartment=row.compartment, family=name.split("_", 1)[0])
            rows.append(result)
        pd.DataFrame(rows).to_csv(OUT / f"radiomics_{rater}.csv", index=False)
    pd.DataFrame(metadata.values()).to_csv(OUT / "feature_metadata.csv", index=False)
    (OUT / "radiomics_parameters.json").write_text(json.dumps({
        "PyRadiomics": radiomics.__version__, "features": FEATURES,
        "binCount_T2": BIN_COUNT_T2, "binWidths": BIN_WIDTHS,
        "normalize": False, "imageType": "Original"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

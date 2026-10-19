from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from shapely.affinity import translate

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "input/wsi_manifest.csv"
OUT = ROOT / "results/wsi"
STAIN_REFERENCE = None  
MAX_ROI_PIXELS = 100_000_000
CLASS_NAMES = {"Tumor": "TC", "Tumor cell": "TC", "TC": "TC",
               "Immune": "TIL", "Lymphocyte": "TIL", "TIL": "TIL",
               "Stroma": "SC", "Stromal": "SC", "SC": "SC",
               "Other": "OC", "OC": "OC"}


def resolve(value):
    p = Path(str(value)); return p if p.is_absolute() else ROOT / p


def features(path):
    obj = json.loads(resolve(path).read_text(encoding="utf-8-sig"))
    return obj["features"] if obj.get("type") == "FeatureCollection" else [obj]


def convert_objects(roi_features, cell_features):
    roi = unary_union([shape(f["geometry"]) for f in roi_features
                       if f.get("geometry") and shape(f["geometry"]).geom_type
                       in {"Polygon", "MultiPolygon"}])
    if roi.is_empty or not roi.is_valid:
        raise ValueError("error")
    left, top = np.floor(roi.bounds[:2]).astype(int)
    right, bottom = np.ceil(roi.bounds[2:]).astype(int)
    cells = []
    for f in cell_features:
        if not f.get("nucleusGeometry"):
            raise ValueError("error")
        nucleus, cell = shape(f["nucleusGeometry"]), shape(f["geometry"])
        if not roi.covers(nucleus.centroid):
            continue
        if (not nucleus.is_valid or not cell.is_valid or
                nucleus.geom_type != "Polygon" or cell.geom_type != "Polygon"):
            raise ValueError("error")
        if cell.area <= 0 or not cell.buffer(1e-6).covers(nucleus):
            raise ValueError("error")
        classification = f.get("properties", {}).get("classification", {})
        name = classification.get("name") if isinstance(classification, dict) else classification
        if name not in CLASS_NAMES:
            raise ValueError("error")
        cells.append(dict(cell_class=CLASS_NAMES[name],
            nucleus=mapping(translate(nucleus, -left, -top)),
            cell=mapping(translate(cell, -left, -top))))
    if not cells:
        raise ValueError("error")
    return translate(roi, -left, -top), cells, (left, top, right, bottom)


def main():
    import openslide
    import tifffile
    import staintools
    if STAIN_REFERENCE is None:
        raise ValueError("error")
    reference = np.asarray(Image.open(resolve(STAIN_REFERENCE)).convert("RGB"))
    normalizer = staintools.StainNormalizer(method="vahadane")
    normalizer.fit(reference)
    table = pd.read_csv(MANIFEST, dtype={"case_id": str, "rater": str})
    if table.duplicated(["case_id", "rater"]).any():
        raise ValueError("error")
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for row in table.itertuples():
        if min(row.mpp_x, row.mpp_y) <= 0:
            raise ValueError("error")
        roi, cells, (left, top, right, bottom) = convert_objects(
            features(row.roi_geojson), features(row.cells_geojson))
        width, height = right-left, bottom-top
        if width*height > MAX_ROI_PIXELS:
            raise MemoryError("error")
        with openslide.OpenSlide(str(resolve(row.slide))) as slide:
            if not (0 <= left < right <= slide.dimensions[0] and
                    0 <= top < bottom <= slide.dimensions[1]):
                raise ValueError("error")
            rgb = np.asarray(slide.read_region((left, top), 0, (width, height)).convert("RGB"))
        normalized = normalizer.transform(rgb)
        folder = OUT / str(row.case_id) / str(row.rater)
        folder.mkdir(parents=True, exist_ok=True)
        image_path, geom_path = folder / "normalized_rgb.tif", folder / "objects.json"
        tifffile.imwrite(image_path, normalized, photometric="rgb")
        geom_path.write_text(json.dumps(dict(roi=mapping(roi), cells=cells,
            mpp_x=float(row.mpp_x), mpp_y=float(row.mpp_y),
            crop_offset_level0=[int(left), int(top)])), encoding="utf-8")
        records.append(dict(case_id=row.case_id, species=row.species, rater=row.rater,
                            image=str(image_path), objects=str(geom_path)))
    pd.DataFrame(records).to_csv(OUT / "processed_manifest.csv", index=False)
    (OUT / "parameters.json").write_text(json.dumps(dict(method="Vahadane",
        stain_reference=str(resolve(STAIN_REFERENCE)), class_mapping=CLASS_NAMES,
        roi_aggregation="geometric_union", resampling=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, cKDTree, distance
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import shape, MultiPoint
from shapely.affinity import scale
from shapely.ops import voronoi_diagram

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "results/wsi/processed_manifest.csv"
OUT = ROOT / "results/features"
DEFINITIONS_CONFIRMED = False
NUCLEAR_LENGTH = None  
ORIENTATION_BIN_DEG = None  
ORIENTATION_RADIUS_UM = None  
RGB_BIN_COUNT = None  
RGB_NORMALIZE = None  
RGB_NORMALIZE_SCALE = 255.
VORONOI_BOUNDARY = "clip_to_roi"  
ORIENTATION_AGGREGATION = "pooled_symmetric_cooccurrence"  


def mean_sd(values, prefix):
    a = np.asarray(values, float)
    return {prefix+"_mean": float(a.mean()) if len(a) else np.nan,
            prefix+"_sd": float(a.std(ddof=1)) if len(a) > 1 else np.nan}


def diameters(poly):
    hull = np.asarray(poly.convex_hull.exterior.coords[:-1])
    maximum = float(distance.pdist(hull).max()) if len(hull) > 1 else 0.
    if len(hull) < 3:
        return maximum, 0.
    edges = np.roll(hull, -1, axis=0)-hull
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.column_stack([-edges[:, 1], edges[:, 0]]) / lengths[:, None]
    projections = hull @ normals.T
    minimum = float(np.ptp(projections, axis=0).min())
    return maximum, minimum


def polygon_axes(poly):
    
    
    rings = [np.asarray(poly.exterior.coords)] + [np.asarray(r.coords) for r in poly.interiors]
    area = sx = sy = sxx = syy = sxy = 0.
    for index, c in enumerate(rings):
        x, y, u, v = c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1]
        cross = x*v-u*y
        sign = (1 if index == 0 else -1) * np.sign(cross.sum())
        cross *= sign
        area += cross.sum()/2
        sx += np.sum((x+u)*cross)/6; sy += np.sum((y+v)*cross)/6
        sxx += np.sum((x*x+x*u+u*u)*cross)/12
        syy += np.sum((y*y+y*v+v*v)*cross)/12
        sxy += np.sum((2*x*y+x*v+u*y+2*u*v)*cross)/24
    if area <= 0:
        return np.nan, np.nan, np.nan
    cx, cy = sx/area, sy/area
    cov = np.array([[sxx/area-cx*cx, sxy/area-cx*cy],
                    [sxy/area-cx*cy, syy/area-cy*cy]])
    values, vectors = np.linalg.eigh(cov)
    axes = 4*np.sqrt(np.maximum(values, 0))
    angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])) % 180
    return float(axes[-1]), float(axes[0]), float(angle)


def cell_features(cells):
    result = {"composition_"+cl: sum(c[0] == cl for c in cells)/len(cells)
              for cl in ["TC", "TIL", "SC", "OC"]}
    measurements = {k: [] for k in ["area", "length", "solidity", "circularity",
                    "maximum_diameter", "minimum_diameter", "nucleus_cell_area_ratio"]}
    points, angles = [], []
    for cls, nucleus, cell in cells:
        if cls != "TC":
            continue
        major, minor, angle = polygon_axes(nucleus)
        maximum, minimum = diameters(nucleus)
        measurements["area"].append(nucleus.area)
        measurements["length"].append(nucleus.length if NUCLEAR_LENGTH == "perimeter" else major)
        measurements["solidity"].append(nucleus.area/nucleus.convex_hull.area)
        measurements["circularity"].append(4*np.pi*nucleus.area/nucleus.length**2)
        measurements["maximum_diameter"].append(maximum)
        measurements["minimum_diameter"].append(minimum)
        measurements["nucleus_cell_area_ratio"].append(nucleus.area/cell.area)
        points.append(nucleus.centroid.coords[0]); angles.append(angle)
    for name, vals in measurements.items():
        result.update(mean_sd(vals, "nuclear_"+name))
    return result, np.asarray(points, float).reshape(-1, 2), np.asarray(angles)


def spatial_features(points, roi):
    r = {}
    vd = {k: [] for k in ["area", "perimeter", "max_distance"]}
    dt = {k: [] for k in ["area", "perimeter"]}
    mst_edges = []
    if len(points) >= 3 and np.linalg.matrix_rank(points-points.mean(0)) == 2:
        regions = voronoi_diagram(MultiPoint(points), envelope=roi.envelope, edges=False)
        for cell in regions.geoms:
            clipped = cell.intersection(roi)
            if not clipped.is_empty and clipped.area > 0:
                vd["area"].append(clipped.area); vd["perimeter"].append(clipped.length)
                vd["max_distance"].append(diameters(clipped.convex_hull)[0])
        triangulation = Delaunay(points)
        edges = set()
        from shapely.geometry import Polygon
        for tri in triangulation.simplices:
            polygon = Polygon(points[tri])
            if roi.covers(polygon):
                dt["area"].append(polygon.area); dt["perimeter"].append(polygon.length)
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]:
                edges.add(tuple(sorted((int(a), int(b)))))
        if edges:
            a, b = np.asarray(sorted(edges)).T
            weights = np.linalg.norm(points[a]-points[b], axis=1)
            graph = coo_matrix((np.r_[weights, weights],
                               (np.r_[a, b], np.r_[b, a])), shape=(len(points),)*2).tocsr()
            mst_edges = minimum_spanning_tree(graph).data
    for name, vals in vd.items(): r.update(mean_sd(vals, "voronoi_"+name))
    for name, vals in dt.items(): r.update(mean_sd(vals, "delaunay_"+name))
    r.update(mean_sd(mst_edges, "mst_edge_length"))
    tree = cKDTree(points) if len(points) else None
    for k in [5, 7]:
        vals = tree.query(points, k=k+1)[0][:, 1:].mean(1) if len(points) > k else []
        r.update(mean_sd(vals, f"density_distance_{k}nn"))
    for radius in [25, 50]:
        vals = np.asarray(tree.query_ball_point(points, radius, return_length=True))-1 if tree else []
        r.update(mean_sd(vals, f"density_count_{radius}um"))
    return r


def orientation_features(points, angles):
    names = ["contrast_energy", "contrast_inverse_moment", "contrast_average",
             "contrast_variance", "contrast_entropy", "intensity_average",
             "intensity_variance", "intensity_entropy", "entropy", "energy", "correlation"]
    if len(points) < 2:
        return {"orientation_"+n: np.nan for n in names}
    pairs = cKDTree(points).query_pairs(ORIENTATION_RADIUS_UM, output_type="ndarray")
    if not len(pairs):
        return {"orientation_"+n: np.nan for n in names}
    bins = int(round(180/ORIENTATION_BIN_DEG))
    labels = np.floor(angles/ORIENTATION_BIN_DEG).astype(int) % bins
    p = np.zeros((bins, bins), float)
    a, b = labels[pairs[:, 0]], labels[pairs[:, 1]]
    np.add.at(p, (a, b), 1); np.add.at(p, (b, a), 1); p /= p.sum()
    i, j = np.indices(p.shape); d = np.abs(i-j); s = i+j
    pdiff = np.bincount(d.ravel(), p.ravel(), minlength=bins)
    psum = np.bincount(s.ravel(), p.ravel(), minlength=2*bins-1)
    def entropy(v):
        v = v[v > 0]; return float(-np.sum(v*np.log2(v)))
    da, sa = float((p*d).sum()), float((p*s).sum())
    pi, pj = p.sum(1), p.sum(0); idx = np.arange(bins)
    mi, mj = pi@idx, pj@idx
    si = np.sqrt(pi@(idx-mi)**2); sj = np.sqrt(pj@(idx-mj)**2)
    corr = float((p*(i-mi)*(j-mj)).sum()/(si*sj)) if si*sj > 0 else np.nan
    vals = [(p*d**2).sum(), (p/(1+d**2)).sum(), da,
            (p*(d-da)**2).sum(), entropy(pdiff), sa,
            (p*(s-sa)**2).sum(), entropy(psum), entropy(p.ravel()), (p*p).sum(), corr]
    return {"orientation_"+n: float(v) for n, v in zip(names, vals)}


def roi_features(rgb, roi, mpp_x, mpp_y):
    import SimpleITK as sitk
    from skimage.draw import polygon as raster_polygon
    from radiomics import firstorder, glcm
    pixel_roi = scale(roi, 1/mpp_x, 1/mpp_y, origin=(0, 0))
    mask = np.zeros(rgb.shape[:2], np.uint8)
    geometries = list(pixel_roi.geoms) if pixel_roi.geom_type == "MultiPolygon" else [pixel_roi]
    for polygon in geometries:
        xy = np.asarray(polygon.exterior.coords); rr, cc = raster_polygon(xy[:, 1], xy[:, 0], mask.shape)
        mask[rr, cc] = 1
        for ring in polygon.interiors:
            xy = np.asarray(ring.coords); rr, cc = raster_polygon(xy[:, 1], xy[:, 0], mask.shape)
            mask[rr, cc] = 0
    
    from skimage.measure import regionprops
    region = regionprops(mask.astype(int), spacing=(mpp_y, mpp_x))[0]
    area, perimeter = roi.area, roi.length
    maximum = diameters(roi.convex_hull)[0]
    major = region.axis_major_length if hasattr(region, "axis_major_length") else region.major_axis_length
    minor = region.axis_minor_length if hasattr(region, "axis_minor_length") else region.minor_axis_length
    circularity = 4*np.pi*area/perimeter**2
    r = {"roi_CircularDisproportion": 1/circularity, "roi_Circularity": circularity,
         "roi_Surface": area, "roi_PixelSurface": int(mask.sum()), "roi_Perimeter": perimeter,
         "roi_PerimeterSurfaceRatio": perimeter/area, "roi_Maximum2DDiameter": maximum,
         "roi_MajorAxisLength": major, "roi_MinorAxisLength": minor,
         "roi_Elongation": minor/major if major else np.nan}
    first_names = ["10Percentile", "90Percentile", "Energy", "Entropy", "InterquartileRange",
                  "Kurtosis", "Maximum", "MeanAbsoluteDeviation", "Mean", "Median", "Minimum",
                  "Range", "RobustMeanAbsoluteDeviation", "RootMeanSquared", "Skewness",
                  "TotalEnergy", "Uniformity", "Variance"]
    glcm_names = ["Autocorrelation", "ClusterProminence", "ClusterShade", "ClusterTendency",
                  "Contrast", "Correlation", "DifferenceAverage", "DifferenceEntropy",
                  "DifferenceVariance", "Id", "Idm", "Idmn", "Idn", "Imc1", "Imc2",
                  "InverseVariance", "JointAverage", "JointEnergy", "JointEntropy", "MCC",
                  "MaximumProbability", "SumAverage", "SumEntropy", "SumSquares"]
    mask_im = sitk.GetImageFromArray(mask); mask_im.SetSpacing((mpp_x, mpp_y))
    for k, channel in enumerate("RGB"):
        image = sitk.GetImageFromArray(rgb[:, :, k].astype(float)); image.SetSpacing((mpp_x, mpp_y))
        if RGB_NORMALIZE:
            vals = rgb[:, :, k][mask > 0].astype(float)
            if vals.std() <= 0: raise ValueError("error")
            arr = (rgb[:, :, k].astype(float)-vals.mean())/vals.std()*RGB_NORMALIZE_SCALE
            image = sitk.GetImageFromArray(arr); image.SetSpacing((mpp_x, mpp_y))
        for family, cls, names in [("firstorder", firstorder.RadiomicsFirstOrder, first_names),
                                   ("glcm", glcm.RadiomicsGLCM, glcm_names)]:
            extractor = cls(image, mask_im, binCount=RGB_BIN_COUNT, label=1)
            extractor.disableAllFeatures()
            for name in names: extractor.enableFeatureByName(name)
            values = extractor.execute()
            for name in names: r[f"{channel}_{family}_{name}"] = float(np.asarray(values[name]))
    return r


def main():
    import tifffile
    if (not DEFINITIONS_CONFIRMED or NUCLEAR_LENGTH not in {"perimeter", "major_axis"}
            or ORIENTATION_BIN_DEG is None or ORIENTATION_RADIUS_UM is None
            or RGB_BIN_COUNT is None or RGB_NORMALIZE is None):
        raise ValueError("error")
    if not np.isclose(180/ORIENTATION_BIN_DEG, round(180/ORIENTATION_BIN_DEG)):
        raise ValueError("error")
    table = pd.read_csv(MANIFEST, dtype={"case_id": str, "rater": str})
    OUT.mkdir(parents=True, exist_ok=True)
    for rater, frame in table.groupby("rater"):
        rows = []
        for row in frame.itertuples():
            obj = json.loads(Path(row.objects).read_text(encoding="utf-8"))
            sx, sy = obj["mpp_x"], obj["mpp_y"]
            physical = lambda g: scale(shape(g), sx, sy, origin=(0, 0))
            roi = physical(obj["roi"])
            cells = [(c["cell_class"], physical(c["nucleus"]), physical(c["cell"]))
                     for c in obj["cells"]]
            r, points, angles = cell_features(cells)
            r.update(spatial_features(points, roi))
            r.update(orientation_features(points, angles))
            r.update(roi_features(tifffile.imread(row.image), roi, sx, sy))
            if len(r) != 185: raise AssertionError("error")
            rows.append(dict(case_id=row.case_id, **r))
        pd.DataFrame(rows).to_csv(OUT / f"pathomics_{rater}.csv", index=False)
    (OUT / "pathomics_parameters.json").write_text(json.dumps(dict(
        nuclear_length=NUCLEAR_LENGTH, orientation_bin_deg=ORIENTATION_BIN_DEG,
        orientation_radius_um=ORIENTATION_RADIUS_UM, orientation_aggregation=ORIENTATION_AGGREGATION,
        rgb_bin_count=RGB_BIN_COUNT, rgb_normalize=RGB_NORMALIZE,
        rgb_normalize_scale=RGB_NORMALIZE_SCALE, voronoi_boundary=VORONOI_BOUNDARY,
        standard_deviation_ddof=1), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

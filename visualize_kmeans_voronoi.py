# -*- coding: utf-8 -*-
# visualize_kmeans_voronoi.py  — robust I/O (Overpass JSON + GeoJSON), fixed min-dist, clearer logs
import os, json, re, argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, shape

# ---------- helpers ----------
def to_number(x):
    if x is None: return np.nan
    if isinstance(x,(int,float)): return float(x)
    s = re.sub(r'[^0-9\-–~.,]', '', str(x)).replace('–','-').replace('~','')
    if '-' in s:
        try:
            a,b = s.split('-',1)
            return (float(a.replace(',','.'))+float(b.replace(',','.')))/2.0
        except: return np.nan
    try: return float(s.replace(',','.'))
    except: return np.nan

def capacity_from_tags(tags: dict):
    tags = tags or {}
    keys = ["beds","capacity:beds","capacity:persons","capacity","rooms","capacity:rooms",
            "capacity:pitches","capacity:tents","capacity:caravans"]
    vals = {k: to_number(tags.get(k)) for k in keys if k in tags}
    for k in ["beds","capacity:beds","capacity:persons","capacity"]:
        if k in vals and not np.isnan(vals[k]): return vals[k]
    if "rooms" in vals and not np.isnan(vals["rooms"]): return vals["rooms"]*2.0
    for k in ["capacity:pitches","capacity:tents","capacity:caravans"]:
        if k in vals and not np.isnan(vals[k]): return vals[k]*3.0
    return np.nan

def weighted_kmeans(points, weights, k, iters=60, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), p=weights/weights.sum())
    centers = [points[idx]]
    for _ in range(1,k):
        d2 = np.min([np.sum((points - c)**2, axis=1) for c in centers], axis=0)
        probs = d2 * weights; probs = probs / probs.sum()
        idx = rng.choice(len(points), p=probs)
        centers.append(points[idx])
    centers = np.array(centers, float)
    for _ in range(iters):
        d2 = np.array([np.sum((points - c)**2, axis=1) for c in centers])
        labels = np.argmin(d2, axis=0)
        new_centers=[]
        for j in range(k):
            m = labels==j
            if not np.any(m):
                idx = rng.choice(len(points), p=weights/weights.sum())
                new_centers.append(points[idx])
            else:
                ww = weights[m]; pp = points[m]
                new_centers.append((pp*ww[:,None]).sum(0)/ww.sum())
        new_centers = np.array(new_centers)
        if np.allclose(new_centers, centers): break
        centers = new_centers
    d2 = np.array([np.sum((points - c)**2, axis=1) for c in centers])
    labels = np.argmin(d2, axis=0)
    return centers, labels

def representatives(points, centers, labels, weights):
    reps=[]
    for j in range(len(centers)):
        m = labels==j
        if not np.any(m): continue
        pts = points[m]
        d2 = np.sum((pts - centers[j])**2, axis=1)
        i = np.argmin(d2)
        reps.append((pts[i,0], pts[i,1], weights[m].sum(), j))
    return np.array(reps) if reps else np.zeros((0,4))

def prune_min_dist(cands_xyw, dmin):
    if cands_xyw.shape[0] == 0: return cands_xyw
    order = np.argsort(cands_xyw[:,2])[::-1]
    keep=[]
    for idx in order:
        x,y,w,_ = cands_xyw[idx]
        ok=True
        for idy in keep:
            ux,uy,uw,_ = cands_xyw[idy]
            # FIXED: was (y-yy)**2; should be (y-uy)**2
            if (x-ux)**2 + (y-uy)**2 < dmin**2:
                ok=False; break
        if ok:
            keep.append(idx)
    return cands_xyw[keep]

def assign_to_sites(points, sites_xy):
    out = np.empty(points.shape[0], dtype=int)
    for i in range(points.shape[0]):
        d2 = np.sum((sites_xy - points[i])**2, axis=1)
        out[i] = int(np.argmin(d2))
    return out

def plot_clusters(points, sites_xy, assign, title, out_png):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:,0], points[:,1], c=assign, s=6, alpha=0.8)
    ax.scatter(sites_xy[:,0], sites_xy[:,1], marker="*", s=140)
    ax.set_title(title)
    ax.set_axis_off()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_with_tg(points, sites_xy, assign, title, out_png, shp_path=None):
    if shp_path is None or not os.path.exists(shp_path):
        return plot_clusters(points, sites_xy, assign, title+" (без контуру ТГ)", out_png)
    try:
        tg = gpd.read_file(shp_path).to_crs(4326).to_crs(3857)
    except Exception:
        return plot_clusters(points, sites_xy, assign, title+" (без контуру ТГ)", out_png)
    fig, ax = plt.subplots(figsize=(8, 8))
    tg.boundary.plot(ax=ax, linewidth=0.5)
    ax.scatter(points[:,0], points[:,1], c=assign, s=6, alpha=0.8)
    ax.scatter(sites_xy[:,0], sites_xy[:,1], marker="*", s=140)
    ax.set_title(title); ax.set_axis_off()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- robust OSM/GeoJSON loader ----------
def load_osm_like(path):
    """
    Returns GeoDataFrame with columns:
      - geometry (Point, EPSG:4326)
      - tags (dict-like, may be empty)
    Accepts Overpass JSON (elements[].tags) or GeoJSON FeatureCollection (features[].properties).
    """
    # Try JSON first
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        obj = None

    rows = []

    # Overpass JSON
    if isinstance(obj, dict) and "elements" in obj:
        for e in obj.get("elements", []):
            tags = e.get("tags", {}) or {}
            if e.get("type") == "node":
                lon, lat = e.get("lon"), e.get("lat")
            else:
                c = e.get("center")
                if not c: continue
                lon, lat = c.get("lon"), c.get("lat")
            if lon is None or lat is None: continue
            rows.append({"tags": tags, "lon": float(lon), "lat": float(lat)})

    # GeoJSON FeatureCollection
    elif isinstance(obj, dict) and obj.get("type") == "FeatureCollection" and "features" in obj:
        for feat in obj.get("features", []):
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry")
            if not geom: continue
            try:
                shp = shape(geom)
                pt = shp.centroid if not shp.geom_type == "Point" else shp
                lon, lat = float(pt.x), float(pt.y)
                rows.append({"tags": props, "lon": lon, "lat": lat})
            except Exception:
                continue

    # Fallback: let GeoPandas read any vector file (geojson/shp/etc.)
    if not rows:
        try:
            g = gpd.read_file(path)
            g = g.to_crs(4326)
            # Build 'tags' from non-geometry columns
            non_geom_cols = [c for c in g.columns if c != "geometry"]
            def row_tags(row):
                return {k: row[k] for k in non_geom_cols if pd.notna(row[k])}
            rows = []
            for _, r in g.iterrows():
                if r.geometry is None: continue
                pt = r.geometry.centroid if r.geometry.geom_type != "Point" else r.geometry
                rows.append({"tags": row_tags(r), "lon": float(pt.x), "lat": float(pt.y)})
        except Exception:
            rows = []

    if not rows:
        raise RuntimeError("Не вдалося прочитати жодної точки з файлу. Перевірте формат/вміст OSM/GeoJSON.")

    gdf = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]), crs=4326)
    return gdf

# ---------- pipeline ----------
def run_for_p(X, w, p, min_dist, out_dir, shp_path=None):
    k = min(p+3, max(3, len(X)))
    centers, labels = weighted_kmeans(X, w, k)
    reps = representatives(X, centers, labels, w)  # (x,y,weight_sum,cluster_id)
    pruned = prune_min_dist(reps, min_dist)
    # adjust count
    if pruned.shape[0] > p:
        pruned = pruned[np.argsort(pruned[:,2])[::-1][:p]]
    elif pruned.shape[0] < p and len(reps)>0:
        need = p - pruned.shape[0]
        candidate_idx = np.argsort(reps[:,2])[::-1]
        existing = {(x,y) for x,y,_,_ in pruned}
        for idx in candidate_idx:
            if need==0: break
            x,y,ww,_ = reps[idx]
            if (x,y) in existing: continue
            ok=True
            for xx,yy,_,_ in pruned:
                if (x-xx)**2 + (y-yy)**2 < min_dist**2:
                    ok=False; break
            if ok:
                pruned = np.vstack([pruned, reps[idx]])
                existing.add((x,y)); need -= 1

    if pruned.shape[0] == 0:
        raise RuntimeError("Після фільтрації мін-відстані не залишилось кандидатів. Зменште --min-dist або перевірте дані.")

    sites_xy = pruned[:, :2]
    assign = assign_to_sites(X, sites_xy)

    png1 = os.path.join(out_dir, f"kmeans_voronoi_{p}.png")
    plot_clusters(X, sites_xy, assign, f"Кластери POI (Voronoi) — p={p}", png1)
    png2 = os.path.join(out_dir, f"kmeans_map_TG_{p}.png")
    plot_with_tg(X, sites_xy, assign, f"Кластери POI + межі ТГ — p={p}", png2, shp_path)

    df = pd.DataFrame({"assign": assign, "w": w})
    summary = df.groupby("assign").agg(n_poi=("assign","size"), sum_weight=("w","sum")).reset_index()
    sites_geo = gpd.GeoSeries([Point(xy) for xy in sites_xy], crs=3857).to_crs(4326)
    summary["lon"] = sites_geo.x; summary["lat"] = sites_geo.y
    csv = os.path.join(out_dir, f"kmeans_voronoi_summary_{p}.csv")
    summary.to_csv(csv, index=False, encoding="utf-8-sig")
    return png1, png2, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="тека з даними (куди писати PNG/CSV)")
    ap.add_argument("--osm", default="osm_tourism_if.geojson", help="шлях до OSM/GeoJSON (Overpass JSON або GeoJSON)")
    ap.add_argument("--shp", default="IF_reg_TG_bou_7.shp", help="необов’язково: межі ТГ (shp) для фону")
    ap.add_argument("--min-dist", type=float, default=2000, help="мінімальна відстань між обраними сайтами (м)")
    args = ap.parse_args()

    osm_path = os.path.join(args.base, args.osm) if not os.path.isabs(args.osm) else args.osm
    shp_path = os.path.join(args.base, args.shp) if not os.path.isabs(args.shp) else args.shp
    assert os.path.exists(osm_path), f"Немає файлу {osm_path}"

    print(f"[INFO] Читаю OSM/GeoJSON: {osm_path}")
    gdf = load_osm_like(osm_path)
    print(f"[INFO] Завантажено POI: {len(gdf)}")

    gdf["cap"] = gdf["tags"].apply(capacity_from_tags)
    gdf["w"] = gdf["cap"].where(gdf["cap"].notna(), 1.0)
    if gdf["w"].max() > 0:
        thr = np.nanpercentile(gdf["w"], 99)
        gdf["w"] = np.minimum(gdf["w"], thr)

    g3857 = gdf.to_crs(3857)
    X = np.vstack([g3857.geometry.x.values, g3857.geometry.y.values]).T
    w = g3857["w"].values.astype(float)

    if len(X) < 3:
        raise RuntimeError("Замало POI (<3). Перевірте фільтри/вміст файлу.")

    for p in (10,12,15):
        out = run_for_p(X, w, p, args.min_dist, args.base, shp_path if os.path.exists(shp_path) else None)
        print(f"[OK] p={p} -> {out}")

if __name__ == "__main__":
    main()

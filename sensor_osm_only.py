# -*- coding: utf-8 -*-
"""
OSM-only LCS siting:
- читає OSM GeoJSON (elements[]) і межі ТГ
- обчислює ваги об’єктів (capacity з OSM tags з імпутацією)
- кластеризує простір ваговим k-means (власна реалізація, без sklearn)
- обирає репрезентативні точки та чистить їх за мін. відстанню
- зберігає три конфігурації: 10, 12, 15 LCS у GeoJSON/CSV

Вхідні файли (лежать у --base):
  osm_tourism_if.geojson
  IF_reg_TG_bou_7.shp (+ .dbf/.shx/.prj/.cpg)

Параметри шейпа:
  --code-col katotth   (як у вас)
  --name-col NAME_UA   (якщо є; інакше буде повтор коду)

Запуск-приклад:
  python sensor_osm_only.py --base E:\work\geo\assets --code-col katotth --name-col NAME_UA --min-dist 2000
"""
import os, json, re, math, argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def to_number(x):
    if x is None: return np.nan
    if isinstance(x,(int,float)): return float(x)
    s = str(x)
    s = re.sub(r'[^0-9\-–~.,]', '', s).replace('–','-').replace('~','')
    if '-' in s:
        try:
            a,b = s.split('-',1); return (float(a.replace(',','.'))+float(b.replace(',','.')))/2.0
        except: return np.nan
    try: return float(s.replace(',','.'))
    except: return np.nan

def capacity_from_tags(tags):
    keys = ["beds","capacity:beds","capacity:persons","capacity","rooms","capacity:rooms",
            "capacity:pitches","capacity:tents","capacity:caravans"]
    vals = {k: to_number(tags.get(k)) for k in keys if k in tags}
    for k in ["beds","capacity:beds","capacity:persons","capacity"]:
        if k in vals and not np.isnan(vals[k]): return vals[k]
    if "rooms" in vals and not np.isnan(vals["rooms"]): return vals["rooms"]*2.0
    for k in ["capacity:pitches","capacity:tents","capacity:caravans"]:
        if k in vals and not np.isnan(vals[k]): return vals[k]*3.0
    return np.nan

def weighted_kmeans(points, weights, k, iters=50, seed=42):
    rng = np.random.default_rng(seed)
    # k-means++ (з вагами)
    idx = rng.choice(len(points), p=weights/weights.sum())
    centers = [points[idx]]
    for _ in range(1,k):
        d2 = np.min([np.sum((points - c)**2, axis=1) for c in centers], axis=0)
        probs = d2 * weights; probs = probs / probs.sum()
        idx = rng.choice(len(points), p=probs)
        centers.append(points[idx])
    centers = np.array(centers, float)
    # ітерації
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
    cl_w = np.array([weights[labels==j].sum() for j in range(k)])
    return centers, labels, cl_w

def pick_reps(points, centers, labels, weights):
    reps=[]
    for j in range(len(centers)):
        m = labels==j
        if not np.any(m): continue
        pts = points[m]
        d2 = np.sum((pts - centers[j])**2, axis=1)
        i = np.argmin(d2)
        reps.append((pts[i,0], pts[i,1], weights[m].sum()))
    return np.array(reps) if reps else np.zeros((0,3))

def prune_min_dist(cands_xyw, dmin):
    order = np.argsort(cands_xyw[:,2])[::-1]
    keep=[]; used=[]
    for idx in order:
        x,y,w = cands_xyw[idx]
        ok=True
        for idy in keep:
            ux,uy,uw = cands_xyw[idy]
            if (x-ux)**2 + (y-uy)**2 < dmin**2:
                ok=False; break
        if ok: keep.append(idx)
    return cands_xyw[keep]

def make_config(X, w, tg, p, min_dist=2000):
    k = min(p+3, max(3, len(X)))
    centers, labels, clw = weighted_kmeans(X, w, k)
    reps = pick_reps(X, centers, labels, w)
    pruned = prune_min_dist(reps, min_dist)
    if pruned.shape[0] > p:
        pruned = pruned[np.argsort(pruned[:,2])[::-1][:p]]
    elif pruned.shape[0] < p:
        # добір із решти (доки дотримується мін. відстань)
        need = p - pruned.shape[0]
        chosen = {(x,y) for x,y,_ in pruned.tolist()}
        for x,y,ww in reps.tolist():
            if need==0: break
            if (x,y) in chosen: continue
            ok=True
            for xx,yy,_ in pruned.tolist():
                if (x-xx)**2 + (y-yy)**2 < (min_dist**2):
                    ok=False; break
            if ok:
                pruned = np.vstack([pruned, np.array([x,y,ww])])
                chosen.add((x,y)); need -= 1
    # повертаємо у lon/lat
    pts = gpd.GeoSeries([Point(xy) for xy in pruned[:,:2]], crs=3857).to_crs(4326)
    out = gpd.GeoDataFrame({"site_id":[f"LCS_{i+1}" for i in range(len(pts))],
                            "weight":pruned[:,2]}, geometry=pts)
    out = gpd.sjoin(out, tg, how="left", predicate="within")
    out = out.rename(columns={"TG_NAME":"tg_name","HKATOTTG":"katottg_code"})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="тека з даними")
    ap.add_argument("--osm", default="osm_tourism_if.geojson")
    ap.add_argument("--shp", default="IF_reg_TG_bou_7.shp")
    ap.add_argument("--code-col", default="katotth")
    ap.add_argument("--name-col", default=None)
    ap.add_argument("--min-dist", type=float, default=2000)
    args = ap.parse_args()

    OSM = os.path.join(args.base, args.osm)
    SHP = os.path.join(args.base, args.shp)
    if not os.path.exists(OSM): raise FileNotFoundError(OSM)
    if not os.path.exists(SHP): raise FileNotFoundError(SHP)

    # OSM
    with open(OSM, "r", encoding="utf-8") as f: osm = json.load(f)
    rows=[]
    for e in osm.get("elements", []):
        tags=e.get("tags",{}) or {}
        if e.get("type")=="node":
            lon,lat=e.get("lon"),e.get("lat")
        else:
            c=e.get("center")
            if not c: continue
            lon,lat=c.get("lon"),c.get("lat")
        rows.append({"name":tags.get("name"), "tags":tags, "lon":lon, "lat":lat})
    gdf = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["lon"] for r in rows],[r["lat"] for r in rows]), crs=4326)
    gdf["cap"] = gdf["tags"].apply(capacity_from_tags)
    gdf["w"] = gdf["cap"].where(gdf["cap"].notna(), 1.0)
    if gdf["w"].max() > 0:
        thr = np.nanpercentile(gdf["w"], 99)
        gdf["w"] = np.minimum(gdf["w"], thr)

    # ТГ
    tg = gpd.read_file(SHP).to_crs(4326)
    code_col = args.code_col
    name_col = args.name_col if args.name_col and args.name_col in tg.columns else code_col
    tg = tg.rename(columns={code_col:"HKATOTTG", name_col:"TG_NAME"})[["HKATOTTG","TG_NAME","geometry"]]

    # координати у метрах
    g3857 = gdf.to_crs(3857)
    X = np.vstack([g3857.geometry.x.values, g3857.geometry.y.values]).T
    w = g3857["w"].values.astype(float)

    for p in (10,12,15):
        cfg = make_config(X, w, tg, p, min_dist=args.min_dist)
        gj = os.path.join(args.base, f"proposed_LCS_{p}.geojson")
        csv = os.path.join(args.base, f"proposed_LCS_{p}.csv")
        cfg[["site_id","weight","katottg_code","tg_name","geometry"]].to_file(gj, driver="GeoJSON")
        pd.DataFrame({
            "site_id": cfg["site_id"],
            "lon": cfg.geometry.x,
            "lat": cfg.geometry.y,
            "weight": cfg["weight"],
            "katottg_code": cfg["katottg_code"],
            "tg_name": cfg["tg_name"]
        }).to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved {p}: {gj} ; {csv}")

if __name__ == "__main__":
    main()

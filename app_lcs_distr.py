# -*- coding: utf-8 -*-
"""
OSM-only LCS siting:
- читає OSM GeoJSON і межі ТГ
- оцінює Counts-KDE і Capacity-KDE
- обирає p LCS (ваговий k-means + мін. відстань)
- пише GeoJSON/CSV + PNG картки
"""

import os, re, json, argparse, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="divide by zero encountered in log")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=".")
ap.add_argument("--osm", default="osm_tourism_if.geojson")
ap.add_argument("--shp", default="IF_reg_TG_bou_7.shp")
ap.add_argument("--code-col", default="katotth", help="поле коду ТГ у шейпі")
ap.add_argument("--name-col", default=None, help="поле назви ТГ у шейпі (якщо None — використовує код)")
ap.add_argument("--p", type=int, default=12, help="кількість LCS")
ap.add_argument("--min-dist", type=float, default=2000, help="мін. відстань між LCS, м")
ap.add_argument("--bw", type=float, default=2000, help="bandwidth KDE, м")
ap.add_argument("--grid", type=float, default=1000, help="крок сітки KDE, м")
ap.add_argument("--capacity-weighted", action="store_true", help="використати Capacity-KDE замість Counts-KDE")
args = ap.parse_args()

BASE = args.base
OSM_JSON = os.path.join(BASE, args.osm)
SHAPE_TG = os.path.join(BASE, args.shp)

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

# 1) load OSM
with open(OSM_JSON, "r", encoding="utf-8") as f:
    osm = json.load(f)
rows=[]
for e in osm.get("elements", []):
    tags=e.get("tags",{}) or {}
    if e.get("type")=="node":
        lon,lat=e.get("lon"),e.get("lat")
    else:
        c=e.get("center"); 
        if not c: continue
        lon,lat=c.get("lon"),c.get("lat")
    rows.append({"name":tags.get("name"), "tags":tags, "lon":lon, "lat":lat})
gdf = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["lon"] for r in rows],[r["lat"] for r in rows]), crs=4326)

# вага як місткість (для Capacity-KDE), інакше 1
gdf["cap"] = gdf["tags"].apply(capacity_from_tags)
gdf["w"] = gdf["cap"].where(gdf["cap"].notna(), 1.0)
if not args.capacity-weighted:
    gdf["w"] = 1.0

# 2) boundaries
tg = gpd.read_file(SHAPE_TG).to_crs(4326)
code_col = args.code_col
name_col = args.name_col if args.name_col and args.name_col in tg.columns else code_col
tg = tg.rename(columns={code_col:"HKATOTTG", name_col:"TG_NAME"})[["HKATOTTG","TG_NAME","geometry"]]
region = tg.to_crs(3857).unary_union

# 3) KDE grid
g3857 = gdf.to_crs(3857)
X = np.vstack([g3857.geometry.x.values, g3857.geometry.y.values]).T
w = g3857["w"].values.astype(float)

minx,miny,maxx,maxy = region.bounds
xs = np.arange(minx, maxx, args.grid)
ys = np.arange(miny, maxy, args.grid)
xx,yy = np.meshgrid(xs,ys)
grid = np.vstack([xx.ravel(), yy.ravel()]).T

kde = KernelDensity(bandwidth=args.bw, kernel="gaussian").fit(X, sample_weight=w)
S = np.exp(kde.score_samples(grid))

# 4) candidate set = OSM точки (доступ/живлення), ваги = локальна KDE
def sample_kde_at(points_xy, grid_xy, values, grid_step):
    # швидкий бінінг на регулярній сітці
    gx = ((points_xy[:,0]-minx)/grid_step).astype(int)
    gy = ((points_xy[:,1]-miny)/grid_step).astype(int)
    idx = gy*(len(xs)) + gx
    idx = np.clip(idx, 0, len(values)-1)
    return values[idx]

cand_xy = X.copy()
cand_kde = sample_kde_at(cand_xy, grid, S, args.grid)
cand_w = (cand_kde * w)  # підсилюємо об’єкти, що стоять у високій KDE і мають вищу місткість

# 5) ваговий k-means (через sample_weight)
k = max(3, min(args.p, len(cand_xy)))
km = KMeans(n_clusters=k, random_state=42, n_init=10)
km.fit(cand_xy, sample_weight=cand_w)
centers = km.cluster_centers_
labels = km.labels_

# обрати представників-кандидатів = найближча реальна точка до центру
cands = []
for i in range(k):
    mask = labels==i
    if not np.any(mask): continue
    pts = cand_xy[mask]
    weights = cand_w[mask]
    center = centers[i]
    d2 = np.sum((pts-center)**2, axis=1)
    j = np.argmin(d2)
    win_xy = pts[j]
    win_w = weights[j]
    cands.append((win_xy[0], win_xy[1], win_w))
cands = np.array(cands)

# 6) мін. відстань (жадібна чистка)
def prune_min_dist(points_xy, weights, dmin):
    keep=[]
    used=[]
    order = np.argsort(weights)[::-1]
    for idx in order:
        x,y = points_xy[idx]
        ok=True
        for (ux,uy) in used:
            if np.hypot(x-ux, y-uy) < dmin:
                ok=False; break
        if ok:
            keep.append(idx)
            used.append((x,y))
    return keep

keep_idx = prune_min_dist(cands[:,:2], cands[:,2], args.min_dist)
if len(keep_idx) > args.p:
    keep_idx = keep_idx[:args.p]

sel_xy = cands[keep_idx,:2]
sel_pts = gpd.GeoSeries([Point(xy) for xy in sel_xy], crs=3857).to_crs(4326)

out = gpd.GeoDataFrame({"site_id":[f"LCS_{i+1}" for i in range(len(sel_pts))]},
                       geometry=sel_pts)

# 7) save results
out_path_geojson = os.path.join(BASE, "proposed_LCS.geojson")
out_path_csv     = os.path.join(BASE, "proposed_LCS.csv")
out.to_file(out_path_geojson, driver="GeoJSON")
df = pd.DataFrame({"site_id":out["site_id"], 
                   "lon":out.geometry.x, "lat":out.geometry.y})
df.to_csv(out_path_csv, index=False, encoding="utf-8-sig")

# 8) quick map
fig,ax = plt.subplots(1,1,figsize=(6,7))
tg.to_crs(3857).boundary.plot(ax=ax, linewidth=0.5, color="k")
ax.scatter(cand_xy[:,0], cand_xy[:,1], s=2)
sel_xy3857 = np.vstack([p.coords[0] for p in out.to_crs(3857).geometry])
ax.scatter(sel_xy3857[:,0], sel_xy3857[:,1], s=60, marker="*", zorder=3)
ax.set_title("LCS sites (OSM-only, capacity-weighted KMeans)")
plt.tight_layout()
plt.savefig(os.path.join(BASE,"proposed_LCS_map.png"), dpi=300, bbox_inches="tight")
print("[OK] Saved:", out_path_geojson, out_path_csv)

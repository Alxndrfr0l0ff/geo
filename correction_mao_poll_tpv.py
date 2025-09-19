# -*- coding: utf-8 -*-
"""
Карти ТПВ (всі підприємства, т/км²) та частка туристичних підприємств у ТПВ (0–1)
+ експорт узагальнювальної таблиці в Excel.

Потрібні файли біля скрипту:
- IF_reg_TG_bou_7.shp (та супутні .dbf/.shx/.prj/.cpg)
- tur_zbir_2019-.xlsx    (TIN/ЄДРПОУ платників турзбору)
- ecol3_cleaned.xlsx     (ТПВ: HKATOTTG, P_YEAR, POLLUTION_VOL, TIN)
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import shapefile  # pip install pyshp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# ------------- налаштування -------------
BASE = Path(".")
SHP = BASE / "IF_reg_TG_bou_7.shp"
YEAR = 2024
STAMP = time.strftime("%Y%m%d_%H%M%S")

# палітри (узгоджені зі стилем попередніх карт)
CMAP_INTENSITY = "YlGnBu"
CMAP_SHARE     = "YlOrBr"

# ------------- читання shapefile -------------
sf = shapefile.Reader(str(SHP))

# атрибути
fields = [f[0] for f in sf.fields if f[0] != "DeletionFlag"]
recs   = [{fields[i]: r[i] for i in range(len(fields))} for r in sf.records()]
attr   = pd.DataFrame(recs)

# обов'язкові поля в шейпі
if "katotth" not in attr.columns or "name_uk" not in attr.columns:
    raise RuntimeError("У шейпфайлі мають бути поля 'katotth' та 'name_uk'.")

# площі з геометрії (як запасний варіант, якщо не зберігали окремим довідником)
# обчислюємо площу полігона у проєкції шейпа (км²).
# Для pyshp беремо тільки зовнішній контур (parts[0]:parts[1]).
def ring_area(coords):
    """Площа полігонального кільця у проєкції шейпа (км² умовних, якщо шейп у метрах — будуть реальні км²)."""
    if len(coords) < 3: 
        return 0.0
    x = np.asarray([p[0] for p in coords], dtype="float64")
    y = np.asarray([p[1] for p in coords], dtype="float64")
    # шуз-формула
    s = np.sum(x*np.roll(y, -1) - y*np.roll(x, -1))
    return float(abs(s) / 2.0) / 1_000_000.0  # якщо одиниці — м² -> км²

shapes = sf.shapes()
areas_calc = []
patches = []
minx = min([s.bbox[0] for s in shapes]); miny = min([s.bbox[1] for s in shapes])
maxx = max([s.bbox[2] for s in shapes]); maxy = max([s.bbox[3] for s in shapes])

for shp in shapes:
    pts = shp.points
    parts = list(shp.parts) + [len(pts)]
    if len(parts) > 1:
        ext = pts[parts[0]:parts[1]]
        patches.append(Polygon(ext, closed=True))
        areas_calc.append(ring_area(ext))
    else:
        patches.append(Polygon(pts, closed=True))
        areas_calc.append(ring_area(pts))

attr["area_km2_geom"] = areas_calc

# катоттг у порядку шарів (щоб зв’язати з patches)
kat_order = [r.get("katotth") for r in recs]

# ------------- турзбір -> перелік TIN туристичних підприємств -------------
tur = pd.read_excel(BASE / "tur_zbir_2019-.xlsx")
tur.columns = [str(c).strip() for c in tur.columns]
# у різних версіях стовпець TIN може називатись по-різному — уніфікуємо:
tin_col = None
for c in tur.columns:
    if c.strip().upper() in {"TIN", "ЄДРПОУ", "EDRPOU"}:
        tin_col = c
        break
if tin_col is None:
    raise RuntimeError("У файлі tur_zbir_2019-.xlsx не знайдено колонки з TIN/ЄДРПОУ.")

tourist_tins = set(
    pd.to_numeric(tur[tin_col], errors="coerce").dropna().astype(np.int64).tolist()
)

# ------------- ТПВ: загальні та «туристичні» -------------
waste = pd.read_excel(BASE / "ecol3_cleaned.xlsx")
waste.columns = [str(c).strip() for c in waste.columns]

# очікувані назви (як у попередніх блоках)
rename_map = {
    "HKATOTTG": "katotth",
    "P_YEAR": "year",
    "POLLUTION_VOL": "value",
    "TIN": "tin",
}
for k, v in list(rename_map.items()):
    if k in waste.columns and v not in waste.columns:
        waste = waste.rename(columns={k: v})

need = {"katotth", "year", "value", "tin"}
missing = need - set(waste.columns)
if missing:
    raise RuntimeError(f"В ecol3_cleaned.xlsx бракує колонок: {missing}")

# числові формати
waste["year"]  = pd.to_numeric(waste["year"], errors="coerce").astype("Int64")
waste["value"] = pd.to_numeric(waste["value"], errors="coerce")
waste["tin"]   = pd.to_numeric(waste["tin"], errors="coerce")

# вибір за роком
dfy = waste[waste["year"] == YEAR].copy()

# агрегації (усього по ТГ та «туристичні» по ТГ)
tot = (dfy.dropna(subset=["katotth"])
          .groupby("katotth", as_index=False)["value"].sum()
          .rename(columns={"value": "waste_msw_total"}))

tou = (dfy[dfy["tin"].isin(tourist_tins)]
          .dropna(subset=["katotth"])
          .groupby("katotth", as_index=False)["value"].sum()
          .rename(columns={"value": "waste_msw_tour"}))

# зведена по ТГ
df = (attr[["katotth", "name_uk", "area_km2_geom"]]
      .merge(tot, on="katotth", how="left")
      .merge(tou, on="katotth", how="left"))

df["waste_msw_total"] = pd.to_numeric(df["waste_msw_total"], errors="coerce").fillna(0.0)
df["waste_msw_tour"]  = pd.to_numeric(df["waste_msw_tour"], errors="coerce").fillna(0.0)

eps = 1e-12
df["waste_msw_per_km2"]   = df["waste_msw_total"] / (df["area_km2_geom"] + eps)
df["waste_msw_share_tour"] = np.where(
    df["waste_msw_total"] > 0, df["waste_msw_tour"] / df["waste_msw_total"], 0.0
)

df["has_data"] = df["waste_msw_total"] > 0
df["is_zero"]  = df["waste_msw_total"] == 0

# для відмальовки у порядку шарів
map_by_kat = {row["katotth"]: row for _, row in df.iterrows()}
vals_intensity = np.array([map_by_kat.get(k, {}).get("waste_msw_per_km2", np.nan) for k in kat_order], dtype=float)
vals_share     = np.array([map_by_kat.get(k, {}).get("waste_msw_share_tour",    np.nan) for k in kat_order], dtype=float)

# ------------- функція побудови карти -------------
def draw_map(values, title, cbar_label, cmap, vmin=None, vmax=None, out_png=None):
    coll = PatchCollection(patches, linewidths=0.35, edgecolor="#999999")
    arr = np.array(values, dtype=float)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        vmin_auto, vmax_auto = 0.0, 1.0
    else:
        lo = float(np.nanpercentile(finite, 1))
        hi = float(np.nanpercentile(finite, 99))
        vmin_auto, vmax_auto = lo, (hi if hi > lo else lo + 1e-9)

    if vmin is None: vmin = vmin_auto
    if vmax is None: vmax = vmax_auto

    coll.set_array(arr)
    coll.set_cmap(cmap)
    coll.set_clim(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8.27, 8.27), dpi=300)
    ax = fig.add_subplot(111)
    ax.add_collection(coll)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nРік: {YEAR}", loc="left", fontsize=12)

    cbar = fig.colorbar(coll, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)

    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ------------- малюнки -------------
out_int_png = BASE / f"map_{YEAR}_msw_intensity_per_km2_{STAMP}.png"
out_shr_png = BASE / f"map_{YEAR}_msw_share_tour_{STAMP}.png"

draw_map(
    vals_intensity,
    "Тверді побутові відходи (усі підприємства) на км²",
    "т/км²",
    CMAP_INTENSITY,
    out_png=out_int_png
)
draw_map(
    vals_share,
    "Частка туристичних підприємств у загальному обсязі ТПВ",
    "частка (0–1)",
    CMAP_SHARE,
    vmin=0.0, vmax=1.0,
    out_png=out_shr_png
)

# ------------- сторінка 2-на-сторінку A4 (опційно) -------------
from PIL import Image, ImageDraw, ImageFont
W, H = 2480, 3508; MARGIN=60; GAP=40
page = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(page)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 40)
except:
    font = ImageFont.load_default()
draw.text((MARGIN, MARGIN), f"ТПВ, {YEAR}: інтенсивність на км² та частка туристичних підприємств", fill="black", font=font)

top = MARGIN + 70; left = MARGIN; right = W - MARGIN; bottom = H - MARGIN
cell_w = (right - left - GAP) // 2; cell_h = (bottom - top)

for i, img_path in enumerate([out_int_png, out_shr_png]):
    im = Image.open(img_path).convert("RGB")
    im_ratio = im.width / im.height; cell_ratio = cell_w / cell_h
    if im_ratio > cell_ratio:
        new_w = cell_w; new_h = int(cell_w / im_ratio)
    else:
        new_h = cell_h; new_w = int(cell_h * im_ratio)
    im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    ox = left + i * (cell_w + GAP) + (cell_w - new_w) // 2
    oy = top + (cell_h - new_h) // 2
    page.paste(im_resized, (ox, oy))

page_png = BASE / f"page_msw_{YEAR}_{STAMP}.png"
page_pdf = BASE / f"page_msw_{YEAR}_{STAMP}.pdf"
page.save(page_png); page.save(page_pdf)

print("Saved PNG:", page_png)
print("Saved PDF:", page_pdf)

# ------------- Excel-вивантаження -------------
xlsx = BASE / f"waste_msw_{YEAR}_by_TG_intensity_{STAMP}.xlsx"
out_tbl = (df[[
    "katotth", "name_uk", "area_km2_geom",
    "waste_msw_total", "waste_msw_per_km2",
    "waste_msw_tour", "waste_msw_share_tour",
    "has_data", "is_zero"
]].rename(columns={
    "area_km2_geom": "area_km2",
    "waste_msw_total": f"waste_msw_total_{YEAR}",
    "waste_msw_per_km2": f"waste_msw_per_km2_{YEAR}",
    "waste_msw_tour": f"waste_msw_tour_{YEAR}",
    "waste_msw_share_tour": f"waste_msw_share_tour_{YEAR}",
    "has_data": f"has_data_{YEAR}",
    "is_zero": f"is_zero_{YEAR}",
}))
with pd.ExcelWriter(xlsx, engine="xlsxwriter") as wr:
    out_tbl.to_excel(wr, index=False, sheet_name="msw_by_TG")
print("Saved XLSX:", xlsx)

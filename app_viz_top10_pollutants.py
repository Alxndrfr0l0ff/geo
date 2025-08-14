# -*- coding: utf-8 -*-
"""
Візуалізація забруднювачів: ТОП-10 (2024), динаміка 2019–2024, карти ТОП-3 (по ТГ).
Автор: EcoGeo

Залежності: pandas, numpy, matplotlib, pyshp (shapefile), pillow, openpyxl/xlsxwriter
Запуск:  python app_viz_top10_pollutants.py
"""

from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
import shapefile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw, ImageFont

# ---------------------- НАЛАШТУВАННЯ ----------------------
BASE = Path(".")
YEAR = 2024
EPS = 1e-9

# Імена файлів
F_AIR = BASE / "ecol1_cleaned.xlsx"
F_WTR = BASE / "ecol2_cleaned.xlsx"
F_MSW = BASE / "ecol3_cleaned.xlsx"
F_D1  = BASE / "d1_dov.xlsx"
F_D2  = BASE / "d2_dov.xlsx"
F_D3  = BASE / "d3_dov.xlsx"
F_DIM = BASE / "tgs_if_pop_area_katottg.xlsx"
F_SHP = BASE / "IF_reg_TG_bou_7.shp"

OUT_XLSX = BASE / "top10_pollutants_2024.xlsx"
OUT_DYN_AIR = BASE / "dyn_air_top10"
OUT_DYN_WTR = BASE / "dyn_water_top10"
OUT_DYN_MSW = BASE / "dyn_msw_top10"
OUT_MAP_AIR = "maps_air_top3"
OUT_MAP_WTR = "maps_water_top3"
OUT_MAP_MSW = "maps_msw_top3"

# ---------------------- ПЕРЕВІРКИ ----------------------
def assert_exists(path: Path, human_name: str):
    if not path.exists():
        sys.exit(f"[Помилка] Не знайдено файл {human_name}: {path}")

for p, name in [
    (F_AIR, "ecol1_cleaned.xlsx"), (F_WTR, "ecol2_cleaned.xlsx"), (F_MSW, "ecol3_cleaned.xlsx"),
    (F_D1, "d1_dov.xlsx"), (F_D2, "d2_dov.xlsx"), (F_D3, "d3_dov.xlsx"),
    (F_SHP, "IF_reg_TG_bou_7.shp"), (F_DIM, "tgs_if_pop_area_katottg.xlsx"),
]:
    assert_exists(p, name)

# ---------------------- ЗАВАНТАЖЕННЯ ДАНИХ ----------------------
def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "P_YEAR" not in df.columns or "HKATOTTG" not in df.columns or "POLLUTION_VOL" not in df.columns:
        miss = {"P_YEAR","HKATOTTG","POLLUTION_VOL"} - set(df.columns)
        sys.exit(f"[Помилка] У вхідному файлі бракує колонок: {miss}")
    df["year"] = pd.to_numeric(df["P_YEAR"], errors="coerce").astype("Int64")
    df["katottg"] = df["HKATOTTG"]
    df["value"] = pd.to_numeric(df["POLLUTION_VOL"], errors="coerce")
    return df

print("[INFO] Читання екологічних файлів ...")
air = prep(pd.read_excel(F_AIR))
wtr = prep(pd.read_excel(F_WTR))
msw = prep(pd.read_excel(F_MSW))

print("[INFO] Читання довідників речовин ...")
d1 = pd.read_excel(F_D1).rename(columns={"Код":"POLLUTION_CODE","Назва":"pollutant_name"})
d2 = pd.read_excel(F_D2).rename(columns={"Код":"POLLUTION_CODE","Назва":"pollutant_name"})
d3 = pd.read_excel(F_D3).rename(columns={"Код":"POLLUTION_CODE","Назва":"pollutant_name"})

print("[INFO] Читання довідника площ ТГ ...")
dim = pd.read_excel(F_DIM).rename(columns={"Код КАТОТТГ":"katottg","Площа, км²":"area_km2"})[["katottg","area_km2"]]
dim["area_km2"] = pd.to_numeric(dim["area_km2"], errors="coerce")

# ---------------------- ТОП-10 ТА ДИНАМІКА ----------------------
def top10_table(df: pd.DataFrame, dov: pd.DataFrame) -> pd.DataFrame:
    d = df[df["year"] == YEAR].dropna(subset=["POLLUTION_CODE","value"]).copy()
    agg = d.groupby("POLLUTION_CODE", as_index=False)["value"].sum().sort_values("value", ascending=False)
    out = agg.merge(dov, on="POLLUTION_CODE", how="left")
    total = out["value"].sum()
    out["share_%"] = (out["value"]/total*100).round(2) if total else 0.0
    cols = ["POLLUTION_CODE","pollutant_name","value","share_%"]
    return out[cols].head(10)

def plot_dynamics(df: pd.DataFrame, top10: pd.DataFrame, dov: pd.DataFrame, title: str, out_stem: Path):
    codes = top10["POLLUTION_CODE"].tolist()
    d = df.dropna(subset=["POLLUTION_CODE","value"]).copy()
    g = d.groupby(["year","POLLUTION_CODE"], as_index=False)["value"].sum()
    g = g[g["POLLUTION_CODE"].isin(codes)]
    wide = g.pivot(index="year", columns="POLLUTION_CODE", values="value").sort_index()
    code2name = dict(zip(dov["POLLUTION_CODE"], dov["pollutant_name"].astype(str)))

    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
    ax = fig.add_subplot(111)
    for c in wide.columns:
        ax.plot(wide.index.astype(int), wide[c], label=code2name.get(c, str(c)))
    ax.set_title(title)
    ax.set_xlabel("Рік"); ax.set_ylabel("Сумарний обсяг (ум. од.)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

print("[INFO] Обчислення ТОП-10 за 2024 р.")
air_top10 = top10_table(air, d1)
wtr_top10 = top10_table(wtr, d2)
msw_top10 = top10_table(msw, d3)

print(f"[INFO] Запис у {OUT_XLSX}")
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as w:
    air_top10.to_excel(w, index=False, sheet_name="Air_2024_TOP10")
    wtr_top10.to_excel(w, index=False, sheet_name="Water_2024_TOP10")
    msw_top10.to_excel(w, index=False, sheet_name="MSW_2024_TOP10")

print("[INFO] Побудова графіків динаміки ...")
plot_dynamics(air, air_top10, d1, "Динаміка ТОП-10 — Атмосфера (2019–2024)", OUT_DYN_AIR)
plot_dynamics(wtr, wtr_top10, d2, "Динаміка ТОП-10 — Вода (2019–2024)", OUT_DYN_WTR)
plot_dynamics(msw, msw_top10, d3, "Динаміка ТОП-10 — ТПВ (2019–2024)", OUT_DYN_MSW)

# ---------------------- КАРТОГРАФІЯ ТОП-3 ----------------------
print("[INFO] Завантаження шейпфайлу ТГ ...")
sf = shapefile.Reader(str(F_SHP))
shapes = sf.shapes()
fields = [f[0] for f in sf.fields if f[0] != "DeletionFlag"]
recs = [{fields[i]: r[i] for i in range(len(fields))} for r in sf.records()]

# Знайти колонку з кодом ТГ у DBF (katotth / katottg / ін.)
def find_code_field(sample: dict) -> str:
    candidates = [k for k in sample.keys() if k.lower() in ("katotth", "katottg")]
    if candidates:
        return candidates[0]
    # якщо не знайдено — спробуємо все в нижньому регістрі на око
    raise RuntimeError(f"Не знайдено поле з кодом ТГ серед полів DBF: {list(sample.keys())}")

code_field = find_code_field(recs[0])
order_kat = [r.get(code_field) for r in recs]

# Межі карти
minx = min([s.bbox[0] for s in shapes]); miny = min([s.bbox[1] for s in shapes])
maxx = max([s.bbox[2] for s in shapes]); maxy = max([s.bbox[3] for s in shapes])
padx = (maxx - minx) * 0.03; pady = (maxy - miny) * 0.03
minx -= padx; maxx += padx; miny -= pady; maxy += pady

# Патчі полігонів (ВАЖЛИВО: closed як іменований аргумент для нових Matplotlib)
patches = []
for shp in shapes:
    pts = shp.points
    parts = list(shp.parts) + [len(pts)]
    # перший контур
    if len(parts) >= 2:
        ring = pts[parts[0]:parts[1]]
        patches.append(Polygon(ring, closed=True))  # <-- КЛЮЧОВЕ ВИПРАВЛЕННЯ

def map_values_for_code(df: pd.DataFrame, code: str, title: str, out_png: Path):
    d = df[(df["year"] == YEAR) & (df["POLLUTION_CODE"] == code)].copy()
    tg = d.groupby("katottg", as_index=False)["value"].sum().rename(columns={"value":"v"})
    grid = dim.merge(tg, on="katottg", how="left")
    grid["v_km2"] = grid["v"] / (grid["area_km2"] + EPS)

    # Впорядкування під запис DBF
    s = grid.set_index("katottg")["v_km2"]
    vals = [s.get(k, np.nan) for k in order_kat]
    arr = np.array(vals, dtype=float)

    finite = arr[np.isfinite(arr)]
    vmin = float(np.nanpercentile(finite, 1)) if finite.size > 0 else 0.0
    vmax = float(np.nanpercentile(finite, 99)) if finite.size > 0 else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    coll = PatchCollection(patches, linewidths=0.35, edgecolor="black")
    coll.set_array(arr)
    coll.set_cmap("YlGnBu")
    coll.set_clim(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8.27, 3.4), dpi=300)  # горизонтальна смуга на А4
    ax = fig.add_subplot(111)
    ax.add_collection(coll)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect('equal', adjustable='box'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10)
    cbar = fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("інтенсивність на км² (ум. од.)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def maps_3in1(df: pd.DataFrame, dov: pd.DataFrame, medium_label: str, base_name: str):
    # ТОП-3 коди за 2024
    top3 = top10_table(df, dov).head(3)[["POLLUTION_CODE","pollutant_name"]]
    pngs = []
    for code, name in zip(top3["POLLUTION_CODE"].tolist(), top3["pollutant_name"].astype(str).tolist()):
        out = BASE / f"{base_name}_{code}.png"
        map_values_for_code(df, code, f"{medium_label} — {name} (ТОП, {YEAR})", out)
        pngs.append(out)

    # Збірка на А4 портрет
    W,H = 2480,3508  # 300 dpi
    MARGIN=50; GAP=35
    page = Image.new("RGB", (W,H), "white")
    draw = ImageDraw.Draw(page)
    # Шрифт
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 44)
    except:
        font = ImageFont.load_default()

    header = f"{medium_label}: ТОП-3 речовини за {YEAR} р.\nІнтенсивність по ТГ, нормовано на площу (ум. од./км²)"
    draw.text((MARGIN, MARGIN), header, fill="black", font=font, spacing=8)

    top_y = MARGIN + 120
    cell_h = (H - top_y - MARGIN - 2*GAP) // 3
    cell_w = W - 2*MARGIN

    for i, fp in enumerate(pngs):
        im = Image.open(fp).convert("RGB")
        im_ratio = im.width / im.height
        cell_ratio = cell_w / cell_h
        if im_ratio > cell_ratio:
            new_w = cell_w; new_h = int(cell_w / im_ratio)
        else:
            new_h = cell_h; new_w = int(cell_h / im_ratio)
        im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        x = MARGIN + (cell_w - new_w)//2
        y = top_y + i*(cell_h + GAP) + (cell_h - new_h)//2
        page.paste(im_resized, (x,y))

    out_png = BASE / f"{base_name}_3in1.png"
    out_pdf = BASE / f"{base_name}_3in1.pdf"
    page.save(out_png); page.save(out_pdf)
    print(f"[OK] Карта 3-в-1: {out_pdf}")

print("[INFO] Побудова карт ТОП-3 ...")
maps_3in1(air, d1, "Атмосферне повітря", OUT_MAP_AIR)
maps_3in1(wtr, d2, "Водні об'єкти",      OUT_MAP_WTR)
maps_3in1(msw, d3, "Тверді побутові відходи", OUT_MAP_MSW)

print("\nГотово.\nФайли створено:")
print(f" - {OUT_XLSX}")
print(f" - {OUT_DYN_AIR}.pdf/.png")
print(f" - {OUT_DYN_WTR}.pdf/.png")
print(f" - {OUT_DYN_MSW}.pdf/.png")
print(f" - {OUT_MAP_AIR}_3in1.pdf/.png")
print(f" - {OUT_MAP_WTR}_3in1.pdf/.png")
print(f" - {OUT_MAP_MSW}_3in1.pdf/.png")
print(f"\nMatplotlib: {matplotlib.__version__}")

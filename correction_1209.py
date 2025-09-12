# -*- coding: utf-8 -*-
import os, datetime
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ====================== КОНФІГ ШЛЯХІВ ======================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(SCRIPT_DIR, "assets")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHAPEFILE       = os.path.join(DATA_DIR, "IF_reg_TG_bou_7.shp")
EMISSIONS_XL    = os.path.join(DATA_DIR, "ecol1_cleaned.xlsx")
EMISSIONS_SHEET = 0
TOUR_TAX_XL     = os.path.join(DATA_DIR, "tur_zbir_2019-.xlsx")
TOUR_TAX_SHEET  = 0

# ================== ФІКСОВАНІ КЛЮЧІ ТА РІК =================
GEOM_KEY      = "katotth"     # поле у шейпі (код ТГ)
TAB_KEY       = "HKATOTTG"    # поле у таблиці викидів (код ТГ)
YEAR_COL      = "Рік"         # якщо немає — поставте None
YEAR_TO_PLOT  = 2024

# ======= ІДЕНТИФІКАТОР ПІДПРИЄМСТВА (у двох таблицях) =======
# якщо у вас інша назва (напр. "ЄДРПОУ") — змініть обидва рядки
EMISSIONS_ID_COL = "TIN"      # у файлі з викидами
TOUR_ID_COL      = "TIN"      # у файлі турзбору

# ===== Яку метрику викидів сумувати (якщо total немає) ======
# Якщо у вашій таблиці є готовий total (напр., "Викиди, т") — вкажіть:
VALUE_COLS = None             # напр.: ["Викиди, т"]; якщо None — сумуються всі числові (крім ключів/року)

# ======================= ПАЛІТРИ (СТИЛЬ) ====================
# фіксовані Brewer-палітри, 6 класів, дискретні
BLUES_6   = ListedColormap(["#eff3ff","#c6dbef","#9ecae1","#6baed6","#3182bd","#08519c"])
ORANGES_6 = ListedColormap(["#feedde","#fdbe85","#fd8d3c","#e6550d","#a63603","#7f2704"])

# форматер — звичайні числа (без 1e+…)
fmt_plain = FuncFormatter(lambda x, pos: f"{x:,.0f}".replace(",", " "))

# ========================= 1) ШЕЙП ==========================
assert os.path.exists(SHAPEFILE), f"Немає шейпфайлу: {SHAPEFILE}"
gdf = gpd.read_file(SHAPEFILE)
assert GEOM_KEY in gdf.columns, f"У шейпі немає '{GEOM_KEY}'"
gdf[GEOM_KEY] = gdf[GEOM_KEY].astype(str).str.strip()

# ================== 2) ТАБЛИЦЯ ВИКИДІВ ======================
assert os.path.exists(EMISSIONS_XL), f"Немає файлу: {EMISSIONS_XL}"
edf = pd.read_excel(EMISSIONS_XL, sheet_name=EMISSIONS_SHEET)
edf.columns = [str(c).strip() for c in edf.columns]
for need in [TAB_KEY, EMISSIONS_ID_COL]:
    assert need in edf.columns, f"У таблиці викидів немає '{need}'. Є: {list(edf.columns)}"

edf[TAB_KEY]          = edf[TAB_KEY].astype(str).str.strip()
edf[EMISSIONS_ID_COL] = edf[EMISSIONS_ID_COL].astype(str).str.strip()

if YEAR_COL and YEAR_COL in edf.columns:
    edf[YEAR_COL] = pd.to_numeric(edf[YEAR_COL], errors="coerce")
    edf = edf[edf[YEAR_COL] == YEAR_TO_PLOT].copy()

if VALUE_COLS is None:
    drop_cols = {TAB_KEY, EMISSIONS_ID_COL}
    if YEAR_COL and YEAR_COL in edf.columns: drop_cols.add(YEAR_COL)
    VALUE_COLS = [c for c in edf.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(edf[c])]
    assert VALUE_COLS, "Не знайдено числових колонок у таблиці викидів. Задайте VALUE_COLS вручну."

for c in VALUE_COLS:
    edf[c] = pd.to_numeric(edf[c], errors="coerce")
edf["emissions_t"] = edf[VALUE_COLS].sum(axis=1, skipna=True)

# ================== 3) ТУРЗБІР (перелік турпідпр.) ==========
assert os.path.exists(TOUR_TAX_XL), f"Немає файлу: {TOUR_TAX_XL}"
tdf = pd.read_excel(TOUR_TAX_XL, sheet_name=TOUR_TAX_SHEET)
tdf.columns = [str(c).strip() for c in tdf.columns]
assert TOUR_ID_COL in tdf.columns, f"У турзборі немає '{TOUR_ID_COL}'. Є: {list(tdf.columns)}"

# фільтр на 2024 рік, якщо у файлі є колонка року
tour_year_cols = [c for c in tdf.columns if c.lower() in ["рік","year","год","rik"]]
if tour_year_cols:
    yy = tour_year_cols[0]
    tdf[yy] = pd.to_numeric(tdf[yy], errors="coerce")
    tdf = tdf[tdf[yy] == YEAR_TO_PLOT].copy()

tdf[TOUR_ID_COL] = tdf[TOUR_ID_COL].astype(str).str.strip()
tour_ids = set(tdf[TOUR_ID_COL].dropna().unique())

# ================== 4) ПОЗНАЧАЄМО «ТУРИЗМ» У ВИКИДАХ ========
edf["is_tour"] = edf[EMISSIONS_ID_COL].isin(tour_ids).astype(int)

# агрегації по ТГ:
tot_by_tg  = edf.groupby(TAB_KEY, as_index=False)["emissions_t"].sum() \
               .rename(columns={"emissions_t":"em_total_tg"})
tour_by_tg = edf[edf["is_tour"]==1].groupby(TAB_KEY, as_index=False)["emissions_t"].sum() \
               .rename(columns={"emissions_t":"em_tour_tg"})

agg = tot_by_tg.merge(tour_by_tg, on=TAB_KEY, how="left").fillna({"em_tour_tg":0})
agg["tour_share"] = np.where(agg["em_total_tg"]>0, agg["em_tour_tg"]/agg["em_total_tg"], 0.0)

# ================== 5) JOIN З ГЕОМЕТРІЄЮ =====================
g = gdf.merge(agg, left_on=GEOM_KEY, right_on=TAB_KEY, how="left") \
       .fillna({"em_total_tg":0, "em_tour_tg":0, "tour_share":0})

# площа, інтенсивність
if g.crs is None: g.set_crs(4326, inplace=True)
if not g.crs.is_projected: g = g.to_crs(32635)
g["area_km2"] = g.geometry.area / 1e6
g["intensity_t_km2"] = np.where(g["area_km2"]>0, g["em_total_tg"]/g["area_km2"], np.nan)

# ================== 6) ДВІ ПАНЕЛІ (СТИЛЬ МАКЕТУ) =============
fig, axes = plt.subplots(1, 2, figsize=(11.2, 6.6), dpi=300, constrained_layout=True)

# --- Ліва: інтенсивність (т/км²), дискретні квантильні пороги (6 класів) ---
valsL = g["intensity_t_km2"].to_numpy()
valsL = valsL[np.isfinite(valsL)]
if len(valsL) >= 6 and np.nanmax(valsL) > 0:
    bins_L = np.unique(np.quantile(valsL, [0, .2, .4, .6, .8, 1.0]))
    # гарантуємо 6 інтервалів (7 порогів)
    if bins_L.size < 7:
        vmax = float(np.nanmax(valsL)); bins_L = np.linspace(0, vmax, 7)
else:
    bins_L = np.linspace(0, 1, 7)
normL = BoundaryNorm(bins_L, BLUES_6.N, clip=True)

g.plot(ax=axes[0], column="intensity_t_km2", cmap=BLUES_6, norm=normL,
       edgecolor="#ffffff", linewidth=0.6, antialiased=True)
axes[0].set_title("Викиди в атмосферу (усі підприємства) на км²\nРік: 2024", fontsize=10)
axes[0].axis("off")
cbarL = plt.cm.ScalarMappable(norm=normL, cmap=BLUES_6)
cbL = fig.colorbar(cbarL, ax=axes[0], fraction=0.035, pad=0.01)
cbL.ax.set_ylabel("т/км²", rotation=90, fontsize=8)
cbL.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
cbL.ax.yaxis.set_major_formatter(fmt_plain)

# --- Права: частка турпідприємств (0..1), фіксовані пороги (6 класів) ---
bins_R = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
normR  = BoundaryNorm(bins_R, ORANGES_6.N, clip=True)

g.plot(ax=axes[1], column="tour_share", cmap=ORANGES_6, norm=normR,
       edgecolor="#ffffff", linewidth=0.6, antialiased=True)
axes[1].set_title("Частка туристичних підприємств у викидах в атмосферу\nРік: 2024", fontsize=10)
axes[1].axis("off")
cbarR = plt.cm.ScalarMappable(norm=normR, cmap=ORANGES_6)
cbR = fig.colorbar(cbarR, ax=axes[1], fraction=0.035, pad=0.01)
cbR.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
cbR.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))

# --- Збереження (PNG + опційно SVG для верстки) ---
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_png = os.path.join(OUTPUT_DIR, f"fig_3_9_style_MATCH_{ts}.png")
plt.savefig(out_png, dpi=300, facecolor="white", bbox_inches="tight")
# також можна розкоментувати SVG:
# out_svg = os.path.join(OUTPUT_DIR, f"fig_3_9_style_MATCH_{ts}.svg")
# plt.savefig(out_svg, facecolor="white", bbox_inches="tight")
print("[OK] Saved:", out_png)

# контрольна таблиця показників:
g.drop(columns="geometry")[["katotth","em_total_tg","em_tour_tg","tour_share","area_km2","intensity_t_km2"]] \
 .to_csv(os.path.join(OUTPUT_DIR, f"fig_3_9_metrics_{ts}.csv"), index=False)

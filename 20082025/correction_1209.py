# -*- coding: utf-8 -*-
# Хороплети викидів: total (т) та інтенсивність (т/км²) по ТГ Івано-Франківщини
# Працює без mapclassify; детальна діагностика та контроль запису виходів.

import os, re, datetime, glob
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# === НАЛАШТУВАННЯ ===
DATA_DIR = r"/assets"         # <- ЗАМІНІТЬ на вашу папку з .shp та .xlsx/.csv
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# === ДОПОМІЖНІ ===
def detect_kat_col(cols):
    cands = [c for c in cols if re.search(r'katott?g', c, re.I)]
    if cands: return cands[0]
    cands = [c for c in cols if re.search(r'kod|code', c, re.I)]
    return cands[0] if cands else None

def detect_name_col(cols):
    for pat in [r'назв', r'name', r'label', r'title', r'громад']:
        cands = [c for c in cols if re.search(pat, c, re.I)]
        if cands: return cands[0]
    return None

def find_shapefile(data_dir):
    pats = glob.glob(os.path.join(data_dir, "IF_reg_TG_bou_7.shp"))
    if pats: return pats[0]
    # fallback: будь-який .shp з TG
    pats = glob.glob(os.path.join(data_dir, "*.shp"))
    return pats[0] if pats else None

def find_emissions_table(data_dir):
    files = []
    for ext in ("*.xlsx","*.xls","*.csv"):
        files += glob.glob(os.path.join(data_dir, ext))
    # пріоритет імовірних файлів
    pref = {"ecol1_cleaned.xlsx":0,"d1_dov.xlsx":1,"d2_dov.xlsx":2,"d3_dov.xlsx":3}
    files = sorted(files, key=lambda p: (pref.get(os.path.basename(p), 99), p.lower()))
    for p in files:
        try:
            if p.lower().endswith(".csv"):
                dfs = {"__csv__": pd.read_csv(p)}
            else:
                xls = pd.ExcelFile(p)
                dfs = {sn: xls.parse(sn) for sn in xls.sheet_names}
        except Exception as e:
            print(f"[WARN] Не можу прочитати {os.path.basename(p)}: {e}")
            continue
        for sn, df in dfs.items():
            df2 = df.copy()
            df2.columns = [str(c).strip() for c in df2.columns]
            kat = detect_kat_col(df2.columns)
            if not kat: 
                continue
            # шукаємо колонки з викидами / забруднювачами
            val_cols = [c for c in df2.columns if re.search(r'викид|emiss|air|повітр|so2|nox|pm|co2|оксид|діоксид|пил', c, re.I)]
            if not val_cols:
                continue
            year_col = next((c for c in df2.columns if re.search(r'year|рік|рiк', c, re.I)), None)
            return dict(path=p, sheet=sn, kat=kat, val_cols=val_cols, year_col=year_col, df=df2)
    return None

# === 1) Геометрія ТГ ===
shp = find_shapefile(DATA_DIR)
assert shp and os.path.exists(shp), f"Не знайдено шейп-файл у {DATA_DIR}"
gdf = gpd.read_file(shp)
geom_key = detect_kat_col(gdf.columns)
name_col = detect_name_col(gdf.columns)
assert geom_key, f"Не знайшов колонку KATOTTG у шейп-файлі. Колонки: {list(gdf.columns)}"

# === 2) Таблиця викидів ===
em = find_emissions_table(DATA_DIR)
assert em, f"Не знайдено підхожої таблиці з викидами у {DATA_DIR}"
df = em["df"].copy()
kat, val_cols, year_col = em["kat"], em["val_cols"], em["year_col"]

# нормалізація типів
df[kat] = df[kat].astype(str).str.strip()
for c in val_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

if year_col:
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

# === 3) Рік для мапи ===
if year_col and df[year_col].notna().any():
    year_to_plot = int(df[year_col].dropna().max())
    dfp = df[df[year_col]==year_to_plot].copy()
else:
    year_to_plot = None
    dfp = df.copy()

dfp["emissions_t"] = dfp[val_cols].sum(axis=1, skipna=True)
agg = (dfp.groupby(kat, as_index=False)["emissions_t"].sum()
       .rename(columns={kat:"KATOTTG"}))

# === 4) Злиття з геометрією ===
g = gdf.copy()
g[geom_key] = g[geom_key].astype(str).str.strip()
g = g.merge(agg, left_on=geom_key, right_on="KATOTTG", how="left")
g["emissions_t"] = g["emissions_t"].fillna(0.0)

# === 5) Площа та інтенсивність ===
if g.crs is None:
    # якщо немає CRS — ставимо WGS84 як дефолт (більшість шейпів у WGS84)
    g.set_crs(4326, inplace=True)
if not g.crs.is_projected:
    # Івано-Франківська обл. — UTM 35N
    g = g.to_crs(32635)

g["area_km2"] = g.geometry.area / 1e6
g["intensity_t_km2"] = g["emissions_t"] / g["area_km2"].replace({0: np.nan})

# === 6) Побудова без схем класифікації (безперервні колірні шкали) ===
tot_png = os.path.join(OUTPUT_DIR, f"choropleth_emissions_total_{TS}.png")
int_png = os.path.join(OUTPUT_DIR, f"choropleth_emissions_intensity_{TS}.png")
csv_out = os.path.join(OUTPUT_DIR, f"choropleth_emissions_join_{TS}.csv")

ax = g.plot(column="emissions_t", legend=True, edgecolor="black", linewidth=0.2)
plt.title(f"Викиди, т" + (f" (за {year_to_plot} р.)" if year_to_plot else ""))
plt.axis("off")
plt.savefig(tot_png, dpi=300, bbox_inches="tight"); plt.close()

ax2 = g.plot(column="intensity_t_km2", legend=True, edgecolor="black", linewidth=0.2)
plt.title(f"Інтенсивність викидів, т/км²" + (f" (за {year_to_plot} р.)" if year_to_plot else ""))
plt.axis("off")
plt.savefig(int_png, dpi=300, bbox_inches="tight"); plt.close()

g.drop(columns="geometry").to_csv(csv_out, index=False)

# === 7) Діагностика: ТОП-5 та Бурштин ===
flat = g.drop(columns="geometry").copy()
top5 = flat.sort_values("emissions_t", ascending=False).head(5)
print("\n[OK] Згенеровані файли:")
print(tot_png)
print(int_png)
print(csv_out)
print("\n[INFO] ТОП-5 ТГ за обсягом викидів (т):")
print(top5[[geom_key, ('Назва громади' if 'Назва громади' in flat.columns else (name_col or geom_key)), "emissions_t"]].to_string(index=False))

if name_col and name_col in flat.columns:
    burs = flat[flat[name_col].astype(str).str.contains("Бурштин", case=False, na=False)]
    if not burs.empty:
        burs_em = float(burs.iloc[0]["emissions_t"])
        rank = int((flat["emissions_t"] > burs_em).sum() + 1)
        print(f"\n[CHECK] '{burs.iloc[0][name_col]}' — {burs_em:.2f} т; ранг за total: #{rank}")
else:
    print("\n[NOTE] Не знайшов текстову колонку з назвою громади — перевірте вручну по коду KATOTTG.")

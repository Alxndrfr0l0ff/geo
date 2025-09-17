# -*- coding: utf-8 -*-
"""
build_maps_air_2024.py
Автор: EcoGeo
Призначення: побудова хороплетних карт для ВИКИДІВ В АТМОСФЕРУ (2024),
             уніфікований стиль (YlGnBu / YlOrBr), без geopandas.
Потрібні пакети: numpy, pandas, pyproj, shapefile (pyshp), matplotlib, pillow
"""

from pathlib import Path
import os, re, time
import numpy as np
import pandas as pd
import shapefile                     # pyshp
from pyproj import Geod
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont

# ------------------ НАЛАШТУВАННЯ ------------------
YEAR = 2024
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "assets"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = time.strftime("%Y%m%d_%H%M%S")

SHAPE_PATH     = DATA_DIR / "IF_reg_TG_bou_7.shp"
ECOL1_XLSX     = DATA_DIR / "ecol1_cleaned.xlsx"
TUR_TINS_XLSX  = DATA_DIR / "tur_zbir_2019-.xlsx"  # перелік суб'єктів, що платили турзбір
# ---------------------------------------------------

# 1) ЧИТАННЯ ДАНИХ ПРО ВИКИДИ (атмосфера) -------------------------------
assert SHAPE_PATH.exists(), f"Не знайдено шейп: {SHAPE_PATH}"
assert ECOL1_XLSX.exists(), f"Не знайдено файл викидів: {ECOL1_XLSX}"

ecol1 = pd.read_excel(ECOL1_XLSX)
# Очікувані назви колонок:
#   HKATOTTG  - код ТГ (КАТОТТГ)
#   P_YEAR    - рік
#   POLLUTION_VOL - обсяг викидів у тоннах
#   TIN       - ідентифікатор платника (ЕГРПОУ/ІПН)
for req in ["HKATOTTG", "P_YEAR", "POLLUTION_VOL"]:
    if req not in ecol1.columns:
        raise ValueError(f"У {ECOL1_XLSX} відсутня колонка '{req}'")

ecol1["P_YEAR"] = pd.to_numeric(ecol1["P_YEAR"], errors="coerce").astype("Int64")
ecol1["POLLUTION_VOL"] = pd.to_numeric(ecol1["POLLUTION_VOL"], errors="coerce")
# TIN може бути відсутнім/строковим → нормуємо
if "TIN" in ecol1.columns:
    ecol1["TIN"] = pd.to_numeric(ecol1["TIN"], errors="coerce").astype("Int64")
else:
    ecol1["TIN"] = pd.Series([pd.NA]*len(ecol1), dtype="Int64")

# 2) ПЕРЕЛІК ТУРИСТИЧНИХ ПЛАТНИКІВ (TIN) ---------------------------------
tour_tins = set()
if TUR_TINS_XLSX.exists():
    tur = pd.read_excel(TUR_TINS_XLSX)
    tur.columns = [str(c).strip() for c in tur.columns]
    if "TIN" in tur.columns:
        tour_tins = set(pd.to_numeric(tur["TIN"], errors="coerce").dropna().astype("Int64").tolist())
    else:
        # м’який пошук за типово вживаними заголовками
        cand = [c for c in tur.columns if c.lower() in ("єдрпоу","ipn","tin","inn","edrpou","код","код_єдрпоу")]
        if cand:
            tour_tins = set(pd.to_numeric(tur[cand[0]], errors="coerce").dropna().astype("Int64").tolist())

# 3) АГРЕГАЦІЯ за ТГ (2024) ----------------------------------------------
eco24 = ecol1[ecol1["P_YEAR"] == YEAR].copy()

# Всього викидів по ТГ
total_24 = (eco24.groupby("HKATOTTG", as_index=False)["POLLUTION_VOL"]
                 .sum().rename(columns={"POLLUTION_VOL": "em_total_t"}))

# Викиди туристичних підприємств
if tour_tins:
    tour_24 = (eco24[eco24["TIN"].isin(list(tour_tins))]
                    .groupby("HKATOTTG", as_index=False)["POLLUTION_VOL"]
                    .sum().rename(columns={"POLLUTION_VOL": "em_tour_t"}))
else:
    tour_24 = total_24.copy()
    tour_24["em_tour_t"] = 0.0

# 4) ШЕЙПФАЙЛ: атрибути + геометрія + fallback-площа ---------------------
sf = shapefile.Reader(str(SHAPE_PATH))

# атрибути (records): очікуємо 'katotth' і 'name_uk'
fields = [f[0] for f in sf.fields if f[0] != "DeletionFlag"]
recs = [{fields[i]: r[i] for i in range(len(fields))} for r in sf.records()]
attr = pd.DataFrame(recs)
if "katotth" not in attr.columns or "name_uk" not in attr.columns:
    raise ValueError("У шейпі мають бути атрибути 'katotth' і 'name_uk'")

# полігони для карти + межі карти + заготовка площі (NaN)
shapes = sf.shapes()
patches = []
xs_all, ys_all = [], []
for shp in shapes:
    pts_all = shp.points
    parts = list(shp.parts) + [len(pts_all)]
    if len(parts) >= 2:
        exterior = pts_all[parts[0]:parts[1]]
        if len(exterior) >= 3:
            patches.append(Polygon(exterior, closed=True))
            xs_all.extend([p[0] for p in exterior])
            ys_all.extend([p[1] for p in exterior])

# fallback-площа: залишаємо NaN (реально беремо з «вікі»-списку)
attr["area_km2_fallback"] = np.nan

# 5) ПЛОЩІ З ВАШОГО «ВІКІ»-СПИСКУ (ПРІОРИТЕТ) ----------------------------
raw_text = """Білоберізька сільська громада 370,9
Більшівцівська селищна громада 153,3
Богородчанська селищна громада 255,5
Болехівська міська громада 244,2
Брошнів-Осадська селищна громада 94
Букачівська селищна громада 142,8
Бурштинська міська громада 203,6
Верхнянська сільська громада 141,7
Верховинська селищна громада 429,4
Вигодська селищна громада 797,8
Витвицька сільська громада 180,2
Войнилівська селищна громада 163,4
Ворохтянська селищна громада 274,2
Галицька міська громада 246,5
Гвіздецька селищна громада 66,1
Городенківська міська громада 622
Делятинська селищна громада 209,5
Дзвиняцька сільська громада 101,8
Долинська міська громада 372,6
Дубівська сільська громада 89,7
Дубовецька сільська громада 175,6
Єзупільська селищна громада 87,3
Заболотівська селищна громада 215
Загвіздянська сільська громада 31,10
Зеленська сільська громада 482,2
Івано-Франківська міська громада 265,7
Калуська міська громада 268,9
Коломийська міська громада 174,7
Коршівська сільська громада 132,5
Косівська міська громада 326,8
Космацька сільська громада 110
Кутська селищна громада 115,5
Ланчинська селищна громада 86,2
Лисецька селищна громада 83,7
Матеївецька сільська громада 108
Надвірнянська міська громада 192,9
Нижньовербізька сільська громада 97,7
Новицька сільська громада 144,4
Обертинська селищна громада 162,3
Олешанська сільська громада 157,1
Отинійська селищна громада 214
Пасічнянська сільська громада 424,5
Перегінська селищна громада 669,6
Переріслянська сільська громада 100,4
Печеніжинська селищна громада 186
Підгайчиківська сільська громада 58,8
Поляницька сільська громада 327,4
Пʼядицька сільська громада 128,3
Рогатинська міська громада 652,6
Рожнівська сільська громада 100,8
Рожнятівська селищна громада 171,4
Снятинська міська громада 369,1
Солотвинська селищна громада 377,6
Спаська сільська громада 252,1
Старобогородчанська сільська громада 86,3
Тисменицька міська громада 249,7
Тлумацька міська громада 367,8
Угринівська сільська громада 18,60
Чернелицька селищна громада 130,4
Яблунівська селищна громада 207,3
Ямницька сільська громада 128,3
Яремчанська міська громада 273,7"""

pairs = []
for line in [l.strip() for l in raw_text.strip().splitlines() if l.strip()]:
    m = re.match(r"(.+?)\s+([\d,\.]+)$", line)
    if m:
        nm = m.group(1).strip()
        area_val = float(m.group(2).replace(",", "."))
        pairs.append((nm, area_val))
areas_wiki = pd.DataFrame(pairs, columns=["name_uk", "area_km2_wiki"])

# 6) ЗЛИТТЯ: назви/коди/площі + викиди + тур-частка ----------------------
# беремо коди/назви/NaN-площу із шейпа
df = attr[["katotth", "name_uk", "area_km2_fallback"]].copy()

# додаємо «вікі»-площі (мають пріоритет)
df = df.merge(areas_wiki, on="name_uk", how="left")

# додаємо викиди (всього) і туристичні
df = (df.merge(total_24, left_on="katotth", right_on="HKATOTTG", how="left")
        .merge(tour_24,  on="HKATOTTG", how="left"))

# колонка з правої таблиці нам більше не потрібна, залишаємо тільки katotth
df.drop(columns=["HKATOTTG"], inplace=True, errors="ignore")

# числові та розрахункові поля
df["em_total_t"] = pd.to_numeric(df["em_total_t"], errors="coerce").fillna(0.0)
df["em_tour_t"]  = pd.to_numeric(df["em_tour_t"],  errors="coerce").fillna(0.0)
df["area_km2"]   = np.where(df["area_km2_wiki"].notna(),
                            df["area_km2_wiki"], df["area_km2_fallback"])

eps = 1e-12
df["intensity_t_km2"] = df["em_total_t"] / (df["area_km2"] + eps)
df["tour_share"]      = np.where(df["em_total_t"] > 0,
                                 df["em_tour_t"] / df["em_total_t"], np.nan)

# контрольна сума (має збігатися з «всього по області» за 2024)
print(f"[INFO] Сума по області (тонн): {df['em_total_t'].sum():,.2f}")


# 7) ПІДГОТОВКА ДО МАЛЮВАННЯ --------------------------------------------
# Порядок shapes() == порядок записів у attr, тому беремо kat-список прямо з attr
kat_list = attr["katotth"].tolist()

# індекс за katotth, але сам стовпець залишимо в таблиці
df_idx = df.set_index("katotth", drop=False)

def grid_values(colname: str) -> np.ndarray:
    vals = []
    for k in kat_list:
        if k in df_idx.index and colname in df_idx.columns:
            vals.append(df_idx.at[k, colname])
        else:
            vals.append(np.nan)
    return np.array(vals, dtype=float)

# 8) СТИЛЬНІ ФУНКЦІЇ ДЛЯ КАРТ -------------------------------------------
minx, maxx = min(xs_all), max(xs_all)
miny, maxy = min(ys_all), max(ys_all)
pad_x = (maxx - minx) * 0.03
pad_y = (maxy - miny) * 0.03
minx -= pad_x; maxx += pad_x; miny -= pad_y; maxy += pad_y


def _auto_limits(vals, lo_pct=1, hi_pct=99, fallback=(0.0, 1.0)):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    lo = float(np.nanpercentile(arr, lo_pct))
    hi = float(np.nanpercentile(arr, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return fallback
    return lo, hi

def draw_map(values, title, cbar_label, out_png, cmap="YlGnBu", vmin=None, vmax=None):
    coll = PatchCollection(patches, linewidths=0.35, edgecolor="black", antialiased=True)
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if vmin is None or vmax is None:
        vmin_auto, vmax_auto = _auto_limits(finite, 1, 99, (0.0, 1.0))
        if vmin is None: vmin = vmin_auto
        if vmax is None: vmax = vmax_auto
    coll.set_array(arr)
    coll.set_cmap(cmap)
    coll.set_clim(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)  # A4 портрет
    ax = fig.add_subplot(111)
    ax.add_collection(coll)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect('equal', adjustable='box'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nРік: {YEAR}", loc="left", fontsize=11)
    cbar = fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    return out_png

def compose_two(img_left, img_right, title_text, filename_base):
    W, H = 2480, 3508  # A4 @300dpi
    MARGIN, GAP = 60, 40
    page = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.text((MARGIN, MARGIN), title_text, fill="black", font=font)
    top = MARGIN + 70; left = MARGIN; right = W - MARGIN; bottom = H - MARGIN
    cell_w = (right - left - GAP)//2; cell_h = (bottom - top)
    for ix, img_path in enumerate([img_left, img_right]):
        im = Image.open(img_path).convert("RGB")
        im_ratio = im.width/im.height; cell_ratio = cell_w/cell_h
        if im_ratio > cell_ratio:
            new_w = cell_w; new_h = int(cell_w/im_ratio)
        else:
            new_h = cell_h; new_w = int(cell_h*im_ratio)
        im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        ox = left + ix*(cell_w + GAP) + (cell_w - new_w)//2
        oy = top + (cell_h - new_h)//2
        page.paste(im_resized, (ox, oy))
    out_png = OUTPUT_DIR / f"{filename_base}_{STAMP}.png"
    out_pdf = OUTPUT_DIR / f"{filename_base}_{STAMP}.pdf"
    page.save(out_png); page.save(out_pdf)
    return out_png, out_pdf

# 9) ПОБУДОВА КАРТ (АТМОСФЕРА) ------------------------------------------
vals_intensity = grid_values("intensity_t_km2")
vals_tourshare = grid_values("tour_share")

png_air_int  = draw_map(
    values     = vals_intensity,
    title      = "Інтенсивність викидів в атмосферу (усі підприємства)",
    cbar_label = "т/км²",
    out_png    = OUTPUT_DIR / f"map_{YEAR}_air_intensity_tkm2_{STAMP}.png",
    cmap       = "YlGnBu"
)

png_air_share = draw_map(
    values     = vals_tourshare,
    title      = "Частка туристичних підприємств у викидах в атмосферу",
    cbar_label = "частка (0–1)",
    out_png    = OUTPUT_DIR / f"map_{YEAR}_air_share_tour_{STAMP}.png",
    cmap       = "YlOrBr",
    vmin       = 0.0,
    vmax       = 1.0
)

# 10) КОМПОНОВАННЯ ПАРИ НА A4 -------------------------------------------
pg_air = compose_two(png_air_int, png_air_share,
                     "АТМОСФЕРА, 2024: інтенсивність на км² та частка туристичних підприємств",
                     "page_air_2024")

# 11) ЗБЕРЕЖЕННЯ ТАБЛИЦІ (EXCEL) ----------------------------------------
out_xlsx = OUTPUT_DIR / f"emissions_air_{YEAR}_by_TG_intensity_{STAMP}.xlsx"
(df[["katotth","name_uk","em_total_t","em_tour_t","area_km2","intensity_t_km2","tour_share"]]
   .sort_values(["intensity_t_km2"], ascending=False)
   .to_excel(out_xlsx, index=False))

print("[OK] Карти:", png_air_int, png_air_share)
print("[OK] A4:", pg_air)
print("[OK] Excel:", out_xlsx)
print("[INFO] Підпис до рис.: «Викиди в атмосферу у територіальних громадах Івано-Франківської області у 2024 р., " \
      "нормовано на площу (т/км²) та частка туристичних підприємств (0–1). " \
      "Джерела: екоподаток 2019–2024; розрахунки автора.»")

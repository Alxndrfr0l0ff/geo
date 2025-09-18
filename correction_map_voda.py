# -*- coding: utf-8 -*-
# =============================== #
#  ВОДОСПОЖИВАННЯ (2024): КАРТИ  #
# =============================== #
from pathlib import Path
import time, re, math, json
import numpy as np
import pandas as pd
import shapefile                    # pip install pyshp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw, ImageFont

# -------------------- Налаштування --------------------
YEAR = 2024
BASE = Path.cwd()                   # корінь проєкту (де лежать скрипт і підпапка assets)
ASSETS = BASE / "assets"
OUTDIR = BASE / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)
SHP_PATH = ASSETS / "IF_reg_TG_bou_7.shp"   # поля: katotth, name_uk
VODA_XLSX = BASE / "voda.xlsx"              # r8 (м3), tin, TG (катоттг)
TOUR_XLSX = BASE / "tur_zbir_2019-.xlsx"    # колонка TIN (код платника)
STAMP = time.strftime("%Y%m%d_%H%M%S")

# Стиль палітр — як у розділі "Атмосфера"
CMAP_INTENS = "YlGnBu"
CMAP_SHARE  = "YlOrBr"

# -------------------- Площі ТГ (з Вікі) --------------------
# Той самий блок, що ми вже узгодили; ключ = назва ТГ як у полі name_uk шейпа
raw_text = """
Білоберізька сільська громада 370,9
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
Яремчанська міська громада 273,7
"""

def parse_area_table(text: str) -> pd.DataFrame:
    rows = []
    for line in [l.strip() for l in text.strip().splitlines() if l.strip()]:
        # останнє число у рядку — площа
        m = re.search(r"(.*)\s+([\d\.,]+)$", line)
        if not m: 
            continue
        name = m.group(1).strip()
        area = float(m.group(2).replace(",", "."))
        rows.append({"name_uk": name, "area_km2": area})
    return pd.DataFrame(rows)

df_area = parse_area_table(raw_text)

# -------------------- Шейп та геометрія --------------------
assert SHP_PATH.exists(), f"Не знайдено шейп: {SHP_PATH}"
sf = shapefile.Reader(str(SHP_PATH))
fields = [f[0] for f in sf.fields if f[0] != "DeletionFlag"]
recs = [{fields[i]: r[i] for i in range(len(fields))} for r in sf.records()]
attr = pd.DataFrame(recs)

# Перевірка необхідних полів
for col in ["katotth", "name_uk"]:
    assert col in attr.columns, f"У шейпі немає обов'язкового поля '{col}'"

# Приєднуємо площі за назвою
attr = attr.merge(df_area, on="name_uk", how="left")

# Геометрія для відмалювання
shapes = sf.shapes()
minx = min(s.bbox[0] for s in shapes); miny = min(s.bbox[1] for s in shapes)
maxx = max(s.bbox[2] for s in shapes); maxy = max(s.bbox[3] for s in shapes)
pad_x = (maxx - minx) * 0.03; pad_y = (maxy - miny) * 0.03
minx -= pad_x; maxx += pad_x; miny -= pad_y; maxy += pad_y

patches = []
for shp in shapes:
    pts_all = shp.points
    parts = list(shp.parts) + [len(pts_all)]
    if len(parts) <= 1:
        continue
    ext = pts_all[parts[0]:parts[1]]
    if len(ext) >= 3:
        patches.append(Polygon(ext, closed=True))

# Порядок ТГ у шейпі (масив katotth для індексації)
h2k = [r.get("katotth") for r in recs]

# -------------------- Дані: вода + туристичні TIN --------------------
def coerce_num(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([np.nan]*len(s))

# туристичні суб'єкти (TIN)
tour_df = pd.read_excel(TOUR_XLSX)
tour_df.columns = [str(c).strip() for c in tour_df.columns]
possible_tin_cols = [c for c in tour_df.columns if c.lower() in ("tin","єдрпоу","edrpou","код","code")]
assert possible_tin_cols, "У файлі tur_zbir_2019-.xlsx не знайдено колонку з TIN/ЄДРПОУ"
TIN_COL_TOUR = possible_tin_cols[0]
tour_tins = coerce_num(tour_df[TIN_COL_TOUR]).dropna().astype("Int64").astype(int)
tour_tins = set(tour_tins.tolist())

# вода
voda = pd.read_excel(VODA_XLSX)
voda.columns = [str(c).strip() for c in voda.columns]

# нормалізуємо імена ключових колонок
col_r8   = [c for c in voda.columns if c.lower() == "r8"]
col_tin  = [c for c in voda.columns if c.lower() in ("tin","єдрпоу","edrpou","code","код")]
col_tg   = [c for c in voda.columns if c.lower() in ("tg","katottg","hkatottg","катоттг","код_громади")]
col_year = [c for c in voda.columns if c.lower() in ("period_year","year","рік")]

assert col_r8,   "У voda.xlsx не знайдено колонку 'r8' (водоспоживання, м³)"
assert col_tin,  "У voda.xlsx не знайдено колонку з TIN"
assert col_tg,   "У voda.xlsx не знайдено колонку 'TG/katottg'"
assert col_year, "У voda.xlsx не знайдено колонку з роком (PERIOD_YEAR/year)"

voda = voda.rename(columns={
    col_r8[0]  : "water_m3",
    col_tin[0] : "TIN",
    col_tg[0]  : "katotth",
    col_year[0]: "year"
})

voda["water_m3"] = pd.to_numeric(voda["water_m3"], errors="coerce")
voda["TIN"]      = pd.to_numeric(voda["TIN"], errors="coerce").astype("Int64")
voda["year"]     = pd.to_numeric(voda["year"], errors="coerce").astype("Int64")

# агрегування 2024: усього та лише туристичні
voda24 = voda[voda["year"] == YEAR].copy()
total_agg = (voda24
             .dropna(subset=["katotth"])
             .groupby("katotth", as_index=False)["water_m3"].sum()
             .rename(columns={"water_m3":"water_total_m3_2024"}))

tour_agg = (voda24[voda24["TIN"].isin(tour_tins)]
            .dropna(subset=["katotth"])
            .groupby("katotth", as_index=False)["water_m3"].sum()
            .rename(columns={"water_m3":"water_tour_m3_2024"}))

# зведення у таблицю ТГ
df = (attr[["katotth","name_uk","area_km2"]]
      .merge(total_agg, on="katotth", how="left")
      .merge(tour_agg,  on="katotth", how="left"))

df["water_total_m3_2024"] = df["water_total_m3_2024"].fillna(0.0)
df["water_tour_m3_2024"]  = df["water_tour_m3_2024"].fillna(0.0)

eps = 1e-12
df["water_per_km2_2024"] = df["water_total_m3_2024"] / (df["area_km2"] + eps)
den = df["water_total_m3_2024"].replace(0, np.nan)
df["share_tour_2024"] = (df["water_tour_m3_2024"] / den).fillna(0.0)

df["has_data_2024"] = df["water_total_m3_2024"].notna()
df["is_zero_2024"]  = df["water_total_m3_2024"].abs() < 1e-9

# контрольні суми
region_total = float(df["water_total_m3_2024"].sum())
region_tour  = float(df["water_tour_m3_2024"].sum())
print(f"[INFO] Вода 2024, всього по області, м³: {region_total:,.0f}".replace(",", " "))
print(f"[INFO] Вода 2024, туристичні TIN, м³:    {region_tour:,.0f}".replace(",", " "))

# -------------------- Експорт Excel --------------------
excel_path = OUTDIR / f"water_use_{YEAR}_by_TG_intensity_{STAMP}.xlsx"
cols_out = ["katotth","name_uk","area_km2",
            "water_total_m3_2024","water_per_km2_2024",
            "water_tour_m3_2024","share_tour_2024",
            "has_data_2024","is_zero_2024"]
df[cols_out].to_excel(excel_path, index=False)
print("XLSX:", excel_path)

# -------------------- Функції для мап --------------------
def array_by_kat(values_by_kat: dict) -> np.ndarray:
    """Повертає масив значень у порядку полігонів шейпа (за katotth)."""
    return np.array([values_by_kat.get(k, np.nan) for k in h2k], dtype=float)

def percentiles_auto(arr: np.ndarray, p_lo=1, p_hi=99):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(finite, p_lo))
    hi = float(np.nanpercentile(finite, p_hi))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi

def add_hatching(ax, shp_obj, hatch="..."):
    """Наносить легку крапкову штриховку поверх полігону."""
    pts_all = shp_obj.points
    parts = list(shp_obj.parts) + [len(pts_all)]
    if len(parts) <= 1:
        return
    ring = pts_all[parts[0]:parts[1]]
    poly = Polygon(ring, closed=True, facecolor="none", edgecolor="none", hatch=hatch, linewidth=0.0)
    ax.add_patch(poly)

def draw_map(values, title, cbar_label, out_png, cmap="YlGnBu", vmin=None, vmax=None, zeros_mask=None):
    coll = PatchCollection(patches, linewidths=0.35, edgecolor="black")
    arr = np.array(values, dtype=float)
    if vmin is None or vmax is None:
        lo, hi = percentiles_auto(arr, 1, 99)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    coll.set_array(arr); coll.set_cmap(cmap); coll.set_clim(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8.27, 11.69), dpi=300)  # A4
    ax = fig.add_subplot(111)
    ax.add_collection(coll)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect('equal', adjustable='box'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nРік: {YEAR}", loc="left", fontsize=11)

    # Легка штриховка для ТГ без даних/нуля
    if zeros_mask is not None and len(zeros_mask) == len(shapes):
        for i, shp_obj in enumerate(shapes):
            if zeros_mask[i]:
                add_hatching(ax, shp_obj, hatch="..")

    cbar = fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def compose_two(img_left, img_right, title_text, filename_base):
    W, H = 2480, 3508; MARGIN=60; GAP=40
    page = Image.new("RGB",(W,H),"white"); draw = ImageDraw.Draw(page)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except: font = ImageFont.load_default()
    draw.text((MARGIN,MARGIN), title_text, fill="black", font=font)
    top = MARGIN + 70; left = MARGIN; right = W - MARGIN; bottom = H - MARGIN
    cell_w = (right - left - GAP)//2; cell_h = (bottom - top)
    for ix, img_path in enumerate([img_left, img_right]):
        im = Image.open(img_path).convert("RGB")
        im_ratio = im.width/im.height; cell_ratio = cell_w/cell_h
        if im_ratio > cell_ratio:
            new_w = cell_w; new_h = int(cell_w/im_ratio)
        else:
            new_h = cell_h; new_w = int(cell_h*im_ratio)
        im_resized = im.resize((new_w,new_h), Image.Resampling.LANCZOS)
        ox = left + ix*(cell_w + GAP) + (cell_w - new_w)//2
        oy = top + (cell_h - new_h)//2
        page.paste(im_resized,(ox,oy))
    out_png = OUTDIR / f"{filename_base}_{STAMP}.png"
    out_pdf = OUTDIR / f"{filename_base}_{STAMP}.pdf"
    page.save(out_png); page.save(out_pdf)
    return out_png, out_pdf

# -------------------- Підготовка сіток значень --------------------
# мапи katotth -> значення
m_intensity = dict(zip(df["katotth"], df["water_per_km2_2024"]))
m_share     = dict(zip(df["katotth"], df["share_tour_2024"]))

vals_intensity = array_by_kat(m_intensity)
vals_share     = array_by_kat(m_share)

# Маска для штриховки (де немає водоспоживання/даних)
mask_zero = []
for i, kat in enumerate(h2k):
    row = df.loc[df["katotth"] == kat]
    if row.empty:
        mask_zero.append(True)   # немає запису — штрихуємо
    else:
        z = float(row["water_total_m3_2024"].values[0])
        mask_zero.append(abs(z) < 1e-9)

# -------------------- Малюємо карти --------------------
png_int = OUTDIR / f"map_{YEAR}_water_intensity_per_km2_{STAMP}.png"
png_shr = OUTDIR / f"map_{YEAR}_water_share_tour_{STAMP}.png"

draw_map(vals_intensity,
         "Водоспоживання (всі підприємства) на км²",
         "м³/км²",
         png_int,
         cmap=CMAP_INTENS,
         vmin=None, vmax=None,
         zeros_mask=mask_zero)

draw_map(vals_share,
         "Частка туристичних підприємств у водоспоживанні",
         "частка (0–1)",
         png_shr,
         cmap=CMAP_SHARE,
         vmin=0.0, vmax=1.0,
         zeros_mask=None)

# Компоновка 2 на сторінку
pg_pair = compose_two(png_int, png_shr,
                      "ВОДА, 2024: інтенсивність на км² та частка туристичних підприємств",
                      "page_water_2024")

print("PNG:", png_int)
print("PNG:", png_shr)
print("PAGE:", pg_pair[0], pg_pair[1])

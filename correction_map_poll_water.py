# -*- coding: utf-8 -*-
# FILE: correction_map_discharge.py
# Викиди у водні об’єкти (скиди), 2024: інтенсивність т/км² і частка туристичних підприємств

from pathlib import Path
import re, time
import numpy as np
import pandas as pd
import shapefile                         # pip install pyshp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw, ImageFont

# ------------ НАЛАШТУВАННЯ ШЛЯХІВ ------------
BASE = Path(".")                         # текуча папка проекту (або замініть на ваш ABSOLUTE path)
SHP  = BASE / "assets" / "IF_reg_TG_bou_7.shp"   # шлях до шейпфайлу
ECOL = BASE / "assets" / "ecol2_cleaned1.xlsx"    # скиди у водойми (очищений файл)
TURZ = BASE / "assets" / "tur_zbir_2019-.xlsx"   # звітність по турзбору (для TIN)
YEAR = 2024
STAMP = time.strftime("%Y%m%d_%H%M%S")

# ------------ ДОВІДНИК ПЛОЩ (км²) ------------
RAW_AREA_TEXT = """
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
""".strip()

def _norm(s: str) -> str:
    if not isinstance(s, str): return s
    return (s.replace("ʼ","'").replace("’","'").replace("ʹ","'").replace("′","'").strip())

def parse_area_map(text: str):
    area_map = {}
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        # останній токен — число з комою/крапкою
        *name_parts, area_str = line.split()
        name = " ".join(name_parts)
        area = float(area_str.replace(",", "."))
        area_map[_norm(name)] = area
    return area_map

AREA_BY_NAME = parse_area_map(RAW_AREA_TEXT)

# ------------ ЗАГРУЗКА ШЕЙПФАЙЛУ ------------
sf = shapefile.Reader(str(SHP))
fields = [f[0] for f in sf.fields if f[0] != "DeletionFlag"]
records = [{fields[i]: r[i] for i in range(len(fields))} for r in sf.records()]
attr = pd.DataFrame(records)
assert "katotth" in attr.columns and "name_uk" in attr.columns, "У shapefile мають бути поля 'katotth' і 'name_uk'."

# прив’язка площ
attr["name_uk_norm"] = attr["name_uk"].apply(_norm)
attr["area_km2"] = attr["name_uk_norm"].map(AREA_BY_NAME).astype(float)

# геометрія для отрисовки
shapes = sf.shapes()
# bbox для фрейму карти
minx = min([s.bbox[0] for s in shapes]); miny = min([s.bbox[1] for s in shapes])
maxx = max([s.bbox[2] for s in shapes]); maxy = max([s.bbox[3] for s in shapes])
pad_x = (maxx - minx) * 0.03; pad_y = (maxy - miny) * 0.03
minx -= pad_x; maxx += pad_x; miny -= pad_y; maxy += pad_y

# патчі з екстер’єрів полігонів
patches = []
for shp in shapes:
    pts_all = shp.points
    parts = list(shp.parts) + [len(pts_all)]
    if len(parts) <= 1:
        continue
    exterior = pts_all[parts[0]:parts[1]]
    patches.append(Polygon(exterior, closed=True))

# відповідність порядку полігонів і katotth у тій самій послідовності атрибутів
kat_list = attr["katotth"].tolist()
name_list = attr["name_uk"].tolist()
area_list = attr["area_km2"].tolist()

# ------------ ТУРИСТИЧНІ TIN (із турзбору) ------------
tur = pd.read_excel(TURZ)
tur.columns = [str(c).strip() for c in tur.columns]
tur_tins = pd.to_numeric(tur["TIN"], errors="coerce").dropna().astype(np.int64).unique().tolist()
tur_tins_set = set(tur_tins)

# ------------ ДАНІ СКИДІВ У ВОДОЙМИ (ecol2) ------------
df = pd.read_excel(ECOL)
df.columns = [str(c).strip() for c in df.columns]

# очікувані колонки: HKATOTTG, P_YEAR, POLLUTION_VOL, TIN
df = df.rename(columns={"HKATOTTG":"katottg","P_YEAR":"year","POLLUTION_VOL":"value"})
df["katottg"] = df["katottg"].astype(str).str.strip()
df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["TIN"] = pd.to_numeric(df.get("TIN"), errors="coerce").astype("Int64")
# агрегування 2024
df24 = df[df["year"] == YEAR].copy()
tot = df24.groupby("katottg", as_index=False)["value"].sum().rename(columns={"value":"discharge_total_2024"})
tour = df24[df24["TIN"].isin(tur_tins_set)] \
           .groupby("katottg", as_index=False)["value"].sum() \
           .rename(columns={"value": "discharge_tour_2024"})

# зведена таблиця по ТГ
out = (pd.DataFrame({"katotth": kat_list, "name_uk": name_list, "area_km2": area_list})
         .merge(tot.rename(columns={"katottg":"katotth"}), on="katotth", how="left")
         .merge(tour.rename(columns={"katottg":"katotth"}), on="katotth", how="left"))

for c in ["discharge_total_2024","discharge_tour_2024"]:
    out[c] = out[c].fillna(0.0)

eps = 1e-12
out["discharge_per_km2_2024"] = out["discharge_total_2024"] / (out["area_km2"] + eps)
out["share_tour_2024"] = out["discharge_tour_2024"] / (out["discharge_total_2024"] + eps)

# ------------ МАПІНГ ЗНАЧЕНЬ ДО ПАТЧІВ ------------
def values_by_order(series_map):
    d = dict(zip(out["katotth"], out[series_map]))
    return np.array([d.get(k, np.nan) for k in kat_list], dtype=float)

vals_intensity = values_by_order("discharge_per_km2_2024")
vals_share     = values_by_order("share_tour_2024")

# ------------ РЕНДЕРИНГ КАРТ ------------
def draw_choropleth(values, title, cbar_label, cmap="YlGnBu", vmin=None, vmax=None, fmt_int=False, out_png=None):
    coll = PatchCollection(patches, linewidths=0.35, edgecolor="#2f2f2f")
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.nanpercentile(finite, 1))
        hi = float(np.nanpercentile(finite, 99))
        if hi <= lo: hi = lo + 1e-6
    if vmin is None: vmin = lo
    if vmax is None: vmax = hi
    coll.set_array(arr); coll.set_cmap(cmap); coll.set_clim(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(9.5, 8.0), dpi=300)
    ax = fig.add_subplot(111)
    ax.add_collection(coll)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nРік: {YEAR}", loc="left", fontsize=13)
    cbar = fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    if fmt_int:
        # формат підписів тис. або без наукового
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:,.0f}".replace(",", " ") for t in ticks])
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# Карти окремо
png_int = BASE / f"map_{YEAR}_discharge_intensity_per_km2_{STAMP}.png"
png_shr = BASE / f"map_{YEAR}_discharge_share_tour_{STAMP}.png"

draw_choropleth(vals_intensity,
                "Скиди у водні об’єкти (всі підприємства) на км²",
                "т/км²", cmap="YlGnBu", out_png=png_int, fmt_int=True)

draw_choropleth(vals_share,
                "Частка туристичних підприємств у скидах у водні об’єкти",
                "частка (0–1)", cmap="YlOrBr", vmin=0.0, vmax=1.0, out_png=png_shr)

# ------------ КОМПОЗИТ A4 (2 карти на сторінку) ------------
W, H = 2480, 3508; MARGIN=60; GAP=40
def compose_two(img_left, img_right, title_text, filename_base):
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
    out_png = BASE / f"{filename_base}_{STAMP}.png"
    out_pdf = BASE / f"{filename_base}_{STAMP}.pdf"
    page.save(out_png); page.save(out_pdf)
    return out_png, out_pdf

page_png, page_pdf = compose_two(png_int, png_shr,
                                 "ВОДНІ СКИДИ, 2024: інтенсивність на км² та частка туристичних підприємств",
                                 "page_discharge_2024")

# ------------ ВИГРУЗКА XLSX З МЕТРИКАМИ ------------
xlsx_path = BASE / f"discharge_water_{YEAR}_by_TG_intensity_{STAMP}.xlsx"
cols = ["katotth","name_uk","area_km2",
        "discharge_total_2024","discharge_tour_2024",
        "discharge_per_km2_2024","share_tour_2024"]
out[cols].sort_values("discharge_per_km2_2024", ascending=False).to_excel(xlsx_path, index=False)

print("OK maps:", png_int, png_shr)
print("PAGE:", page_pdf)
print("XLSX:", xlsx_path)

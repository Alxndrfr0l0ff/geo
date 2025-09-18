from pathlib import Path
import time, re
import numpy as np
import pandas as pd
import shapefile                     # pyshp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

# --------- ПАРАМЕТРИ ----------
BASE = Path("C:/workz/geo")          # корінна папка з даними
SHP  = BASE / "assets" / "IF_reg_TG_bou_7.shp"
CSV_TOUR = BASE / "assets" / "згруповано_турзбір.csv"
YEAR = 2024
STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT_DIR = BASE / "output"; OUT_DIR.mkdir(exist_ok=True, parents=True)

# --------- ПЛОЩІ ТГ (із вашого «вікі»-списку) ----------
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
""".strip()

def parse_areas(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        # відділяємо назву від числа праворуч
        m = re.match(r"^(.*\S)\s+([\d\.,]+)$", line)
        if not m: continue
        name = m.group(1)
        area = float(m.group(2).replace(",", "."))
        rows.append({"name_uk": name, "area_km2_wiki": area})
    return pd.DataFrame(rows)

areas_wiki = parse_areas(raw_text)

# --------- ШЕЙПФАЙЛ (pyshp) ----------
sf = shapefile.Reader(str(SHP))
fields = [f[0] for f in sf.fields if f[0] != "DeletionFlag"]
recs = [{fields[i]: r[i] for i in range(len(fields))} for r in sf.records()]
attr = pd.DataFrame(recs)  # очікуємо 'katotth', 'name_uk'
if "katotth" not in attr.columns or "name_uk" not in attr.columns:
    raise ValueError("У шейпі мають бути атрибути 'katotth' і 'name_uk'")

# Геометрія + patches
shapes = sf.shapes()
patches_fill = []          # нормальні (із даними)
patches_nodata = []        # громади без турист. навантаження (0 / NaN)
xs_all, ys_all = [], []

# збережемо порядок katotth у тому ж порядку, що й shapes/records
kat_list = attr["katotth"].tolist()

for shp in shapes:
    pts_all = shp.points
    parts = list(shp.parts) + [len(pts_all)]
    if len(parts) >= 2:
        ring = pts_all[parts[0]:parts[1]]
        if len(ring) >= 3:
            xs_all.extend([p[0] for p in ring]); ys_all.extend([p[1] for p in ring])
            # сам патч додамо пізніше після класифікації (маємо знати, чи є дані)
            # тимчасово просто тримаємо координати
            pass

# --------- ДАНІ ТУРПОПИТУ (туристо-добові) ----------
tour = pd.read_csv(CSV_TOUR, encoding="utf-8-sig")
tour = tour.rename(columns={
    "Код_громади":"katotth",
    "Рік":"year",
    "Всього_туристо_діб":"tourist_nights"
})
tour["year"] = pd.to_numeric(tour["year"], errors="coerce").astype("Int64")
tour["tourist_nights"] = pd.to_numeric(tour["tourist_nights"], errors="coerce")

# агрегуємо за рік і TG
tour_agg = (tour.dropna(subset=["katotth","year"])
                .groupby(["katotth","year"], as_index=False)["tourist_nights"].sum())

# --------- ЗЛИТТЯ: площа + туризм ----------
df = attr[["katotth","name_uk"]].merge(areas_wiki, on="name_uk", how="left")  # додаємо площу
df = df.merge(tour_agg[tour_agg["year"]==YEAR][["katotth","tourist_nights"]],
              on="katotth", how="left")

# інтенсивність: туристо-добові / км²
eps = 1e-12
df["tourist_nights"] = pd.to_numeric(df["tourist_nights"], errors="coerce")
df["area_km2"] = pd.to_numeric(df["area_km2_wiki"], errors="coerce")
df["tour_intensity_per_km2"] = df["tourist_nights"] / (df["area_km2"] + eps)

# --------- ПІДГОТОВКА МАСИВІВ ДЛЯ КАРТИ ----------
# масив значень у тому ж порядку, що і записи шейпфайла
df_idx = df.set_index("katotth", drop=False)
vals = []
for k in kat_list:
    v = np.nan
    if k in df_idx.index:
        v = df_idx.at[k, "tour_intensity_per_km2"]
    vals.append(v)
vals = np.array(vals, dtype=float)

# межі карти
minx, maxx = min(xs_all), max(xs_all)
miny, maxy = min(ys_all), max(ys_all)
pad_x = (maxx - minx) * 0.03; pad_y = (maxy - miny) * 0.03
minx -= pad_x; maxx += pad_x; miny -= pad_y; maxy += pad_y

# діапазон кольорів (1–99 перцентилі, як у атмосфері)
finite = vals[np.isfinite(vals)]
if finite.size == 0:
    vmin, vmax = 0.0, 1.0
else:
    vmin = float(np.nanpercentile(finite, 1))
    vmax = float(np.nanpercentile(finite, 99))
    if vmax <= vmin: vmax = vmin + 1e-6

# --------- ПОБУДОВА ПАТЧІВ З РОЗПОДІЛОМ НА «є дані» / «немає» ----------
# повторно ідемо по shapes і паралельно беремо значення
patches_data, patches_nodata = [], []
for shp, val in zip(shapes, vals):
    pts_all = shp.points
    parts = list(shp.parts) + [len(pts_all)]
    if len(parts) >= 2:
        ring = pts_all[parts[0]:parts[1]]
        if len(ring) < 3:
            continue
        poly = Polygon(ring, closed=True)
        # нуль/NaN → у шар "немає даних / 0"
        if not np.isfinite(val) or abs(val) < 1e-12:
            patches_nodata.append(poly)
        else:
            patches_data.append(poly)

# --------- МАЛЮВАННЯ (та сама стилістика, що для атмосфери) ----------
fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
ax = fig.add_subplot(111)

# 1) Штриховані (або крапкові) «без даних / 0»
if patches_nodata:
    coll0 = PatchCollection(patches_nodata,
                            facecolor="#D9D9D9",  # світло-сірий
                            edgecolor="black",
                            linewidths=0.35,
                            hatch="..",           # крапочки
                            alpha=1.0)
    ax.add_collection(coll0)

# 2) Основний шар з даними (градієнт як у атмосфері)
if patches_data:
    # робимо масив тільки з «даних» у правильному порядку
    vals_data = np.array([v for v in vals if np.isfinite(v) and abs(v) >= 1e-12], dtype=float)
    coll = PatchCollection(patches_data, linewidths=0.35, edgecolor="black")
    coll.set_array(vals_data)
    coll.set_cmap("YlGnBu")         # як у карти атмосфери
    coll.set_clim(vmin=vmin, vmax=vmax)
    ax.add_collection(coll)
    cbar = fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Туристо-добові на 1 км²", fontsize=9)

# рамки карти
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_aspect('equal', adjustable='box'); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Туристичне навантаження (туристо-добові на 1 км²)\nРік: {YEAR}",
             loc="left", fontsize=11)

# легенда для «немає даних / 0»
legend_handles = [Patch(facecolor="#D9D9D9", edgecolor="black", hatch="..", label="0 або відсутні дані")]
ax.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=True)

fig.tight_layout()

png_out = OUT_DIR / f"map_{YEAR}_tour_intensity_per_km2_{STAMP}.png"
pdf_out = OUT_DIR / f"map_{YEAR}_tour_intensity_per_km2_{STAMP}.pdf"
fig.savefig(png_out, bbox_inches="tight"); fig.savefig(pdf_out, bbox_inches="tight"); plt.close(fig)

# --------- Компоновка на сторінку A4 (одна карта) ----------
W,H = 2480,3508; MARGIN=60
page = Image.new("RGB",(W,H),"white"); draw = ImageDraw.Draw(page)
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 40)
except:
    font = ImageFont.load_default()
title = f"ТУРИЗМ, {YEAR}: інтенсивність на км²"
draw.text((MARGIN,MARGIN), title, fill="black", font=font)

# Вставляємо карту по центру
im = Image.open(png_out).convert("RGB")
cell_w = W - 2*MARGIN; cell_h = H - 2*MARGIN - 80
im_ratio = im.width/im.height; cell_ratio = cell_w/cell_h
if im_ratio > cell_ratio:
    new_w = cell_w; new_h = int(cell_w/im_ratio)
else:
    new_h = cell_h; new_w = int(cell_h*im_ratio)
im_resized = im.resize((new_w,new_h), Image.Resampling.LANCZOS)
ox = (W - new_w)//2; oy = MARGIN + 70 + (cell_h - new_h)//2
page.paste(im_resized,(ox,oy))

page_png = OUT_DIR / f"page_tour_{YEAR}_{STAMP}.png"
page_pdf = OUT_DIR / f"page_tour_{YEAR}_{STAMP}.pdf"
page.save(page_png); page.save(page_pdf)

print("OK:", str(page_pdf))

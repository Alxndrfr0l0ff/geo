# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
import shapefile  # pyshp

# === ШЛЯХИ (підлаштуй за потреби) ===
BASE = Path(r"C:/workz/geo")              # корінь твого проєкту
SHP  = BASE / "assets" / "IF_reg_TG_bou_7.shp"
CSV  = BASE / "assets" / "згруповано_турзбір.csv"
OUT  = BASE / "assets" / "tourist_nights_2024_by_TG_intensity.xlsx"
YEAR = 2024

# === 1) katotth + name_uk з шейпа ===
sf = shapefile.Reader(str(SHP))
fields = [f[0] for f in sf.fields if f[0] != "DeletionFlag"]
attr = pd.DataFrame([{fields[i]: r[i] for i in range(len(fields))} for r in sf.records()])
attr = attr.rename(columns={"katotth": "katotth", "name_uk": "name_uk"})
attr["katotth"] = attr["katotth"].astype(str).str.strip()
attr["name_uk"] = attr["name_uk"].astype(str).str.strip()

# === 2) площі ТГ (км²) з попередньо зібраного списку з Вікі (твій raw_text) ===
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

def parse_areas(txt):
    rows = []
    for line in [l.strip() for l in txt.splitlines() if l.strip()]:
        name, val = line.rsplit(" ", 1)
        rows.append({"name_uk": name, "area_km2": float(val.replace(",", "."))})
    return pd.DataFrame(rows)

areas = parse_areas(raw_text)
dim = attr.merge(areas, on="name_uk", how="left")

# === 3) Туристо-доби 2024 з CSV ===
tour = pd.read_csv(CSV, encoding="utf-8-sig").rename(
    columns={"Код_громади":"katotth", "Рік":"year", "Всього_туристо_діб":"tourist_nights"}
)
tour["katotth"] = tour["katotth"].astype(str).str.strip()
tour["year"] = pd.to_numeric(tour["year"], errors="coerce").astype("Int64")
tour["tourist_nights"] = pd.to_numeric(tour["tourist_nights"], errors="coerce")

agg24 = (tour[tour["year"] == YEAR]
         .groupby("katotth", as_index=False)["tourist_nights"].sum()
         .rename(columns={"tourist_nights":"tourist_nights_total_2024"}))

# === 4) Зведення та інтенсивність ===
df = dim.merge(agg24, on="katotth", how="left")
df["has_data_2024"] = df["tourist_nights_total_2024"].notna()
df["is_zero_2024"] = df["tourist_nights_total_2024"].fillna(0).eq(0)
eps = 1e-12
df["tourist_nights_per_km2_2024"] = (
    df["tourist_nights_total_2024"].fillna(0) / (df["area_km2"].replace(0, np.nan) + eps)
)

out_cols = [
    "katotth","name_uk","area_km2",
    "tourist_nights_total_2024","tourist_nights_per_km2_2024",
    "has_data_2024","is_zero_2024"
]
df[out_cols].to_excel(OUT, index=False)

print(f"[OK] Saved: {OUT}")
print("Preview:")
print(df[out_cols].head(10))

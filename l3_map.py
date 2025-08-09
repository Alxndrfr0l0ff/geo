# l3_map.py
# -*- coding: utf-8 -*-
"""
Карта розподілу туристичних підприємств по тергромадах (L3, пропорційна)
+ горизонтальна бар‑діаграма ТОП‑10 громад праворуч.
Працює через pyshp (shapefile), без geopandas.

Приклад запуску:
  python l3_map.py --excel kved_dist.xlsx --shp IF_reg_TG_bou_7.shp \
    --code-field katotth --name-field name_uk \
    --cmap YlGnBu --scale log --figw 18 --figh 9 --out l3_map.png
"""

import argparse
import pandas as pd
import shapefile  # pyshp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def build_map(
    excel_path: str,
    sheet: str | None,
    shp_path: str,
    code_field: str,
    name_field: str,
    cmap_name: str,
    scale: str,
    figw: float,
    figh: float,
    out_path: str,
):
    # --- 1) Дані з Excel ---
    df = pd.read_excel(
        excel_path,
        sheet_name=None if sheet in (None, "None", "") else sheet
    )
    if isinstance(df, dict):
        df = df[next(iter(df))]  # перший лист, якщо sheet не задано
    if "L3" not in df.columns:
        raise SystemExit("У файлі Excel немає колонки 'L3'.")
    l3_counts = df["L3"].value_counts()
    l3_counts_dict = l3_counts.to_dict()

    # --- 2) Шейпфайл ---
    sf = shapefile.Reader(shp_path)
    fields = [f[0] for f in sf.fields[1:]]  # без DeletionFlag
    if code_field not in fields:
        raise SystemExit(f"Поле '{code_field}' не знайдено у шейпі. Доступні: {fields}")
    if name_field not in fields:
        raise SystemExit(f"Поле '{name_field}' не знайдено у шейпі. Доступні: {fields}")

    code_idx = fields.index(code_field)
    name_idx = fields.index(name_field)

    patches, values = [], []
    per_comm = {}  # сума по кожній громаді (назва -> кількість)
    for sr in sf.shapeRecords():
        rec = sr.record
        code = rec[code_idx]
        name = rec[name_idx] or str(code)
        count = int(l3_counts_dict.get(code, 0))
        per_comm[name] = int(per_comm.get(name, 0) + count)

        shp = sr.shape
        parts = list(shp.parts) + [len(shp.points)]
        for i in range(len(parts) - 1):
            pts = shp.points[parts[i]:parts[i + 1]]
            patches.append(MplPolygon(pts, closed=True))
            values.append(count)

    values = np.array(values, dtype=float)

    # --- 3) Фігура з фіксованими областями осей (жорсткий лейаут) ---
    fig = plt.figure(figsize=(figw, figh))
    # [left, bottom, width, height] у частках від фігури
    ax_map = fig.add_axes([0.05, 0.10, 0.52, 0.82])   # карта ліворуч (52% ширини)
    ax_bar = fig.add_axes([0.62, 0.12, 0.33, 0.78])   # бар‑чарт праворуч, гарантований зазор

    # --- 4) Карта (пропорційна) ---
    if scale.lower() == "log":
        norm = LogNorm(vmin=max(values[values > 0].min(), 1), vmax=values.max())
    else:
        norm = Normalize(vmin=values.min(), vmax=values.max())

    coll = PatchCollection(
        patches, cmap=cmap_name, norm=norm, edgecolor="black", linewidths=0.35
    )
    coll.set_array(values)
    ax_map.add_collection(coll)
    ax_map.autoscale_view()
    ax_map.set_aspect("equal", adjustable="datalim")  # збереження пропорцій
    ax_map.margins(0.02)
    ax_map.axis("off")
    ax_map.set_title("Розподіл туристичних підприємств по тергромадах", pad=8)

    # Компактний колорбар як inset всередині карти
    cax = inset_axes(ax_map, width="55%", height="4%", loc="lower left", borderpad=1.0)
    cb = plt.colorbar(coll, cax=cax, orientation="horizontal")
    cb.set_label("Кількість підприємств", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # --- 5) ТОП‑10 бар‑діаграма (повні назви) ---
    top_items = sorted(per_comm.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [nm for nm, _ in top_items][::-1]
    vals = [cnt for _, cnt in top_items][::-1]

    ax_bar.barh(range(len(vals)), vals)
    ax_bar.set_yticks(range(len(vals)), labels=names, fontsize=11)
    ax_bar.set_xlabel("К-сть", fontsize=11)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.35)
    ax_bar.set_xlim(0, max(vals) * 1.15)  # запас для чисел праворуч
    for i, v in enumerate(vals):
        ax_bar.text(v, i, f" {v}", va="center", fontsize=11)
    ax_bar.set_title("ТОП‑10 громад за кількістю зареєстрованих туристичних підприємств", pad=8)

    # ВАЖЛИВО: не використовуємо bbox_inches='tight', щоб не зрушувало осі
    plt.savefig(out_path, dpi=240, facecolor="white")
    print(f"Збережено: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--excel", required=True, help="Шлях до kved_dist.xlsx")
    p.add_argument("--sheet", default=None, help="Назва листа Excel (або залиште None)")
    p.add_argument("--shp", required=True, help="Шлях до IF_reg_TG_bou_7.shp")
    p.add_argument("--code-field", default="katotth", help="Поле коду громади у шейпі")
    p.add_argument("--name-field", default="name_uk", help="Поле назви громади у шейпі")
    p.add_argument("--cmap", default="YlGnBu", help="Colormap (YlGnBu, OrRd, viridis тощо)")
    p.add_argument("--scale", default="linear", choices=["linear", "log"], help="Масштаб кольору")
    p.add_argument("--figw", type=float, default=18.0, help="Ширина фігури (дюйми)")
    p.add_argument("--figh", type=float, default=9.0, help="Висота фігури (дюйми)")
    p.add_argument("--out", default="l3_map.png", help="Вихідний PNG")
    args = p.parse_args()

    build_map(
        args.excel, args.sheet, args.shp,
        args.code_field, args.name_field, args.cmap,
        args.scale, args.figw, args.figh, args.out
    )
plt.show()
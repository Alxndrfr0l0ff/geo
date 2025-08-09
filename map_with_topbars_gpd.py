# map_with_topbars_gpd.py
# -*- coding: utf-8 -*-
"""
Карта розподілу туристичних підприємств по тергромадах (geopandas, пропорційна)
+ праворуч горизонтальна бар-діаграма ТОП‑10 громад.
Джерела:
  - Excel з колонкою L3 (коди громад для підприємств)
  - Shapefile з полями коду громади (напр. 'katotth') і назви (напр. 'name_uk')
"""

import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main(excel, sheet, shp, code_field, name_field,
         cmap, scale, figw, figh, out_png):
    # ---- 1) Дані ----
    df = pd.read_excel(excel, sheet_name=None if sheet in (None, "None", "") else sheet)
    if isinstance(df, dict):
        df = df[next(iter(df))]

    if "L3" not in df.columns:
        raise SystemExit("У Excel немає колонки 'L3' (коди тергромад).")

    # підрахунок підприємств по L3
    l3_counts = df["L3"].value_counts().rename("count").to_frame()

    # ---- 2) Геодані ----
    gdf = gpd.read_file(shp)

    if code_field not in gdf.columns:
        raise SystemExit(f"Поле '{code_field}' не знайдено у shapefile. Доступні: {list(gdf.columns)}")
    if name_field not in gdf.columns:
        raise SystemExit(f"Поле '{name_field}' не знайдено у shapefile. Доступні: {list(gdf.columns)}")

    # джойн: геометрія + count
    gdf = gdf.merge(l3_counts, left_on=code_field, right_index=True, how="left")
    gdf["count"] = gdf["count"].fillna(0).astype(int)

    # агрегат по назві громади для ТОП‑10
    top = (gdf.groupby(name_field)["count"].sum()
           .sort_values(ascending=False).head(10))
    top_names = list(top.index[::-1])
    top_vals  = list(top.values[::-1])

    # ---- 3) Макет: пропорційна карта + бари праворуч ----
    fig = plt.figure(figsize=(figw, figh), constrained_layout=False)

    # фіксована розкладка, щоб НІЧОГО не наїжджало
    # [left, bottom, width, height] у частках від фігури
    ax_map = fig.add_axes([0.05, 0.10, 0.52, 0.82])   # карта ~52% ширини
    ax_bar = fig.add_axes([0.62, 0.12, 0.33, 0.78])   # бари, чіткий зазор

    # норма кольору
    vals = gdf["count"].to_numpy()
    if scale.lower() == "log":
        # vmin не може бути 0 у LogNorm
        vmin = max(1, int(vals[vals > 0].min()) if (vals > 0).any() else 1)
        norm = LogNorm(vmin=vmin, vmax=max(vals.max(), vmin))
    else:
        norm = Normalize(vmin=vals.min(), vmax=vals.max())

    # малюємо карту (geopandas сам ставить aspect='equal')
    gdf.plot(column="count", cmap=cmap, linewidth=0.4, edgecolor="black",
             norm=norm, ax=ax_map)
    ax_map.set_aspect('equal')         # дубляж надійності
    ax_map.set_axis_off()
    ax_map.set_title("Розподіл туристичних підприємств по тергромадах", pad=8)

    # кольорбар — компактний inset всередині карти, знизу
    cax = inset_axes(ax_map, width="55%", height="4%", loc="lower left", borderpad=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Кількість підприємств", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ---- 4) ТОП‑10 бар‑діаграма ----
    ax_bar.barh(range(len(top_vals)), top_vals)
    ax_bar.set_yticks(range(len(top_vals)), labels=top_names, fontsize=11)
    ax_bar.set_xlabel("К-сть", fontsize=11)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.35)
    ax_bar.set_xlim(0, max(top_vals) * 1.15)  # запас під числа праворуч
    for i, v in enumerate(top_vals):
        ax_bar.text(v, i, f" {v}", va="center", fontsize=11)
    ax_bar.set_title("ТОП‑10 громад (бар‑діаграма)", pad=8)

    # збереження без bbox_inches='tight', щоб не зміщувало осі
    plt.savefig(out_png, dpi=240, facecolor="white")
    print(f"Збережено: {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Шлях до kved_dist.xlsx (містить колонку L3)")
    ap.add_argument("--sheet", default=None, help="Назва листа Excel (або залишити None)")
    ap.add_argument("--shp", required=True, help="Шлях до IF_reg_TG_bou_7.shp")
    ap.add_argument("--code-field", default="katotth", help="Поле з кодом громади у shapefile")
    ap.add_argument("--name-field", default="name_uk", help="Поле з назвою громади у shapefile")
    ap.add_argument("--cmap", default="YlGnBu", help="Colormap (YlGnBu, OrRd, viridis тощо)")
    ap.add_argument("--scale", default="linear", choices=["linear","log"], help="Масштаб кольору")
    ap.add_argument("--figw", type=float, default=18.0, help="Ширина фігури")
    ap.add_argument("--figh", type=float, default=9.0, help="Висота фігури")
    ap.add_argument("--out", default="l3_map.png", help="Вихідний PNG")
    args = ap.parse_args()

    main(args.excel, args.sheet, args.shp, args.code_field, args.name_field,
         args.cmap, args.scale, args.figw, args.figh, args.out)

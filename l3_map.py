# map_with_topbars_gpd.py
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl

def kfmt(x, _):
    # формат тисяч без десяткових
    return f"{int(x):,}".replace(",", " ")

def main(excel, sheet, shp, code_field, name_field,
         cmap, scale, figw, figh, out_png):

    # ------- дані -------
    df = pd.read_excel(excel, sheet_name=None if sheet in (None, "None", "") else sheet)
    if isinstance(df, dict): df = df[next(iter(df))]
    if "L3" not in df.columns:
        raise SystemExit("У Excel немає колонки 'L3' (коди тергромад).")

    counts = df["L3"].value_counts().rename("count").to_frame()

    gdf = gpd.read_file(shp)
    if code_field not in gdf.columns: raise SystemExit(f"Немає поля '{code_field}' у шейпі.")
    if name_field not in gdf.columns: raise SystemExit(f"Немає поля '{name_field}' у шейпі.")

    gdf = gdf.merge(counts, left_on=code_field, right_index=True, how="left")
    gdf["count"] = gdf["count"].fillna(0).astype(int)

    top = (gdf.groupby(name_field)["count"].sum()
           .sort_values(ascending=False).head(10))
    top_names = list(top.index[::-1])
    top_vals  = list(top.values[::-1])

    vals = gdf["count"].to_numpy()
    if scale == "log":
        vmin = max(1, int(vals[vals > 0].min()) if (vals > 0).any() else 1)
        norm = LogNorm(vmin=vmin, vmax=max(vals.max(), vmin))
    else:
        norm = Normalize(vmin=vals.min(), vmax=vals.max())

    # ------- фігура/осі (жорстка розкладка без перекриттів) -------
    fig = plt.figure(figsize=(12, 12))
    # карта: ліва, 60% ширини, великі поля зверху/знизу
    ax_map   = fig.add_axes([0.06, 0.16, 0.58, 0.78])
    # бар‑чарт: права панель, гарантований зазор
    ax_bar   = fig.add_axes([0.68, 0.18, 0.28, 0.72])
    # колорбар під картою, по центру
    cax      = fig.add_axes([0.22, 0.08, 0.26, 0.035])

    # ------- карта -------
    gdf.plot(column="count", cmap=cmap, norm=norm,
             linewidth=0.35, edgecolor="#2f2f2f", ax=ax_map)
    ax_map.set_aspect('equal')
    ax_map.set_axis_off()
    ax_map.set_title("Розподіл туристичних підприємств (суміжні види діяльності) по тергромадах",
                     fontsize=14, pad=8)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Кількість підприємств", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # ------- бар‑діаграма -------
    ax_bar.barh(range(len(top_vals)), top_vals, height=0.75)
    ax_bar.set_yticks(range(len(top_vals)), labels=top_names, fontsize=10)
    ax_bar.set_xlabel("К-сть", fontsize=11)
    ax_bar.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax_bar.grid(axis="x", linestyle="--", alpha=0.35)
    # запас справа під числа
    ax_bar.set_xlim(0, max(top_vals) * 1.18)
    for i, v in enumerate(top_vals):
        ax_bar.text(v, i, f" {kfmt(v,None)}", va="center", fontsize=10)
    # підчистити рамку
    for spine in ["top","right","left"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.tick_params(axis='y', length=0)
    ax_bar.set_title("ТОП‑10 громад", fontsize=13, pad=6)

    # важливо: без bbox_inches='tight'
    plt.savefig(out_png, dpi=240, facecolor="white")
    print(f"Збережено: {out_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="kved_dist.xlsx (має колонку L3)")
    ap.add_argument("--sheet", default=None, help="Назва листа або None")
    ap.add_argument("--shp", required=True, help="IF_reg_TG_bou_7.shp")
    ap.add_argument("--code-field", default="katotth")
    ap.add_argument("--name-field", default="name_uk")
    ap.add_argument("--cmap", default="YlGnBu")  # OrRd, viridis, cividis
    ap.add_argument("--scale", choices=["linear","log"], default="linear")
    ap.add_argument("--figw", type=float, default=18.0)
    ap.add_argument("--figh", type=float, default=9.5)
    ap.add_argument("--out", default="l3_map.png")
    args = ap.parse_args()
    main(args.excel, args.sheet, args.shp, args.code_field, args.name_field,
         args.cmap, args.scale, args.figw, args.figh, args.out)

# Re-run the annotation pipeline
import pandas as pd
import numpy as np
from math import atan2, degrees

base = "/mnt/data"
files = [
    f"{base}/kmeans_voronoi_summary_10.csv",
    f"{base}/kmeans_voronoi_summary_12.csv",
    f"{base}/kmeans_voronoi_summary_15.csv",
]

anchors = [
    ("Поляниця", 24.49, 48.352, "курортне ядро"),
    ("Яремче", 24.56, 48.45, "курортне ядро"),
    ("Ворохта", 24.55, 48.29, "курортне ядро"),
    ("Івано-Франківськ", 24.71, 48.92, "міський фон"),
    ("Коломия", 25.04, 48.52, "міський фон"),
    ("Калуш", 24.37, 49.02, "промисловий фон"),
    ("Бурштин", 24.63, 49.25, "промисловий фон"),
    ("Косів", 25.10, 48.30, "коридор/долина «під вітром»"),
    ("Верховина", 24.81, 48.15, "висотний бекґраунд"),
]

def nearest_anchor_role(lon, lat):
    best = None; bestd = 1e9
    for name, alon, alat, role in anchors:
        d = (lon-alon)**2 + (lat-alat)**2
        if d < bestd:
            bestd = d; best = (name, role)
    return best

def sector_label(cx, cy, x, y):
    ang = degrees(np.arctan2(y-cy, x-cx))
    if ang < 0: ang += 360
    bins = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360.0]
    labels = ["східний", "північно-східний", "північний", "північно-західний",
              "західний", "південно-західний", "південний", "південно-східний", "східний"]
    for b, lab in zip(bins, labels):
        if ang <= b: return lab
    return "сектор"

def build_comment(role, anchor_name, n_poi, sum_w, sector):
    sw = f"{sum_w:,.0f}".replace(",", " ")
    base = f"Центр тяжіння кластера ({n_poi} POI; сумарна місткість ≈ {sw} ум. місць). "
    if role == "курортне ядро":
        why = f"Контроль пікових навантажень та долинних інверсій у районі {anchor_name}; зменшує відстань до найближчого датчика у {sector} секторі курортної долини."
    elif role == "міський фон":
        why = f"Репрезентативний міський фон (прибуття/логістика, комунікаційна видимість); відокремлення туристичних епізодів від фонових коливань у {anchor_name}."
    elif role == "промисловий фон":
        why = f"Контроль техногенного внеску (SO₂/NOₓ/PM) у {anchor_name}; дає референт для калібрування LCS і відсікання промислових епізодів."
    elif role == "висотний бекґраунд":
        why = f"Регіональний/висотний фон поблизу {anchor_name}; верифікація переносу з долин і перевірка модельних рядів."
    elif role == "коридор/долина «під вітром»":
        why = f"Контроль переносу з курортного ядра вздовж транспортно-вітрового коридору ({sector} сектор); раннє виявлення епізодів."
    else:
        why = f"Балансування покриття у {sector} секторі; зменшення середньої відстані від POI до найближчого датчика."
    return base + why

outputs = []
previews = {}

for path in files:
    df = pd.read_csv(path)
    expected = {"assign","n_poi","sum_weight","lon","lat"}
    if not expected.issubset(set(df.columns)):
        raise RuntimeError(f"{path}: очікувані колонки {expected}, наявні {set(df.columns)}")
    df["rank_cap"] = df["sum_weight"].rank(ascending=False, method="min").astype(int)
    cx, cy = df["lon"].mean(), df["lat"].mean()
    roles=[]; hints=[]; sectors=[]; comments=[]
    for _, r in df.iterrows():
        name, role = nearest_anchor_role(r["lon"], r["lat"])
        roles.append(role); hints.append(name)
        sec = sector_label(cx, cy, r["lon"], r["lat"]); sectors.append(sec)
        comments.append(build_comment(role, name, int(r["n_poi"]), float(r["sum_weight"]), sec))
    df["role"] = roles
    df["anchor_hint"] = hints
    df["sector"] = sectors
    df["comment"] = comments
    df = df.sort_values(["rank_cap","assign"]).reset_index(drop=True)
    out_csv = path.replace(".csv", "_annotated.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    outputs.append(out_csv)
    previews[out_csv] = df.head(12)  # preview up to 12 rows

from caas_jupyter_tools import display_dataframe_to_user
for out_csv, preview in previews.items():
    display_dataframe_to_user(f"Попередній перегляд: {out_csv}", preview)

outputs

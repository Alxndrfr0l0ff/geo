# ecogeo_pipeline.py
# Бібліотеки: pip install pandas numpy shapefile matplotlib pillow statsmodels
from pathlib import Path
import pandas as pd, numpy as np, shapefile, math, warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw, ImageFont
import statsmodels.api as sm
warnings.filterwarnings("ignore")

BASE = Path(".")
YEAR_RANGE = list(range(2019, 2025))
YEAR = 2024

# ---------- 1) Читання шейпу і площ ----------
sf = shapefile.Reader(str(BASE / "IF_reg_TG_bou_7.shp"))
shapes = sf.shapes()
fields = [f[0] for f in sf.fields if f[0]!="DeletionFlag"]
recs = [dict(zip(fields, r)) for r in sf.records()]
code_field = "katottg" if "katottg" in fields else "katotth"
name_field = "name"
codes = [r[code_field] for r in recs]; names = [r[name_field] for r in recs]
tg_df = pd.DataFrame({"katottg": codes, "name": names})

def norm_apo(s): return s.replace("'", "ʼ").replace("’","ʼ").replace("`","ʼ")
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
rows = []
for line in [l.strip() for l in raw_text.splitlines() if l.strip()]:
    nm, ar = line.rsplit(" ",1); rows.append((nm, float(ar.replace(",","."))))
areas = pd.DataFrame(rows, columns=["name","area_km2"])
tg_df["name_norm"] = tg_df["name"].apply(norm_apo)
areas["name_norm"] = areas["name"].apply(norm_apo)
areas = tg_df.merge(areas.drop(columns=["name"]), on="name_norm", how="left")[["katottg","name","area_km2"]]

# ---------- 2) Дані 2019–2024 і інтенсивності ----------
tour = pd.read_csv(BASE / "згруповано_турзбір.csv")
tn = tour.groupby(["Рік","Код_громади"])["Всього_туристо_діб"].sum().reset_index()\
         .rename(columns={"Рік":"year","Код_громади":"katottg","Всього_туристо_діб":"TN"})
voda = pd.read_excel(BASE / "voda.xlsx")
voda["sumW"] = voda[["R7","R8","R8_1","R8_2"]].fillna(0).sum(axis=1)
w = voda.groupby(["PERIOD_YEAR","TG"])["sumW"].sum().reset_index()\
        .rename(columns={"PERIOD_YEAR":"year","TG":"katottg","sumW":"W"})
def agg_ecol_all(path, newname):
    df = pd.read_excel(path); df["year"] = pd.to_numeric(df["P_YEAR"], errors="coerce")
    return df.groupby(["year","HKATOTTG"])["POLLUTION_VOL"].sum().reset_index()\
             .rename(columns={"HKATOTTG":"katottg","POLLUTION_VOL":newname})
air = agg_ecol_all(BASE/"ecol1_cleaned.xlsx","AIR")
dw  = agg_ecol_all(BASE/"ecol2_cleaned.xlsx","DW")
msw = agg_ecol_all(BASE/"ecol3_cleaned.xlsx","MSW")

years = pd.DataFrame({"year": YEAR_RANGE})
panel = (areas.assign(key=1).merge(years.assign(key=1), on="key").drop(columns=["key"])
         .merge(tn, on=["katottg","year"], how="left")
         .merge(w,  on=["katottg","year"], how="left")
         .merge(air,on=["katottg","year"], how="left")
         .merge(dw, on=["katottg","year"], how="left")
         .merge(msw,on=["katottg","year"], how="left"))
for col in ["TN","W","AIR","DW","MSW"]:
    panel[col] = panel[col].fillna(0.0)
for c in ["TN","W","AIR","DW","MSW"]:
    panel[f"{c}_km2"] = panel[c]/panel["area_km2"]

# Вибірка 2024 (для LISA-карт)
grid = panel[panel["year"]==YEAR].reset_index(drop=True)

# ---------- 3) Ваги kNN і «queen-approx» ----------
def poly_centroid(pts):
    if len(pts)<3:
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; return (sum(xs)/len(xs), sum(ys)/len(ys))
    a=cx=cy=0.0
    for i in range(len(pts)):
        x1,y1=pts[i]; x2,y2=pts[(i+1)%len(pts)]
        cross=x1*y2-x2*y1; a+=cross; cx+=(x1+x2)*cross; cy+=(y1+y2)*cross
    a*=0.5
    if abs(a)<1e-12:
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]; return (sum(xs)/len(xs), sum(ys)/len(ys))
    return (cx/(6*a), cy/(6*a))

patches=[]; centroids=[]; bboxes=[]
for shp in shapes:
    pts=shp.points; parts=list(shp.parts)+[len(pts)]
    ring=pts[parts[0]:parts[1]] if len(parts)>=2 else pts
    patches.append(Polygon(ring, closed=True)); centroids.append(poly_centroid(ring)); bboxes.append(shp.bbox)
XY = np.array(centroids); n=len(XY)

def kNN_weights(XY,k):
    W = np.zeros((n,n),float)
    for i in range(n):
        d = np.sqrt(np.sum((XY-XY[i])**2, axis=1))
        idx=np.argsort(d); neigh=[j for j in idx if j!=i][:k]
        for j in neigh: W[i,j]=1.0
    row=W.sum(axis=1,keepdims=True); row[row==0]=1.0
    return W/row
W4 = kNN_weights(XY,4); W5 = kNN_weights(XY,5)

Wq = np.zeros((n,n),float)
for i in range(n):
    minx1,miny1,maxx1,maxy1 = bboxes[i]
    for j in range(n):
        if i==j: continue
        minx2,miny2,maxx2,maxy2 = bboxes[j]
        inter = not (maxx1<minx2 or maxx2<minx1 or maxy1<miny2 or maxy2<miny1)
        if inter: Wq[i,j]=1.0
row=Wq.sum(axis=1,keepdims=True); row[row==0]=1.0
Wq = Wq/row

# ---------- 4) Глобальний і локальний Моран ----------
def moran_global(y, W, nperm=999, seed=2025):
    rng=np.random.default_rng(seed); y=np.asarray(y,float); z=y-y.mean(); z2=(z*z).sum()
    S0=W.sum(); I=(len(y)/S0)*(z@(W@z))/(z2 if z2>0 else 1.0)
    dist=[]
    for _ in range(nperm):
        zp=rng.permutation(z)
        dist.append((len(y)/S0)*(zp@(W@zp))/((zp*zp).sum()))
    dist=np.array(dist); p=(np.sum(np.abs(dist)>=abs(I))+1)/(nperm+1)
    return I,p

def local_moran(y, W, nperm=999, seed=2025):
    rng=np.random.default_rng(seed); y=np.asarray(y,float)
    z=(y-y.mean())/(y.std(ddof=1) if y.std(ddof=1)>0 else 1.0); lag=W@z; Ii=z*lag
    perm_I=np.zeros((nperm,len(y)))
    for p in range(nperm):
        zp=rng.permutation(z); perm_I[p,:]=z*(W@zp)
    pvals=(np.sum(np.abs(perm_I)>=np.abs(Ii),axis=0)+1)/(nperm+1)
    return Ii,pvals,z,lag

def fdr_bh(pvals, alpha=0.05):
    p=np.asarray(pvals); n=len(p); idx=np.argsort(p); crit=(np.arange(1,n+1)/n)*alpha
    thresh=0.0
    for r,i in enumerate(idx, start=1):
        if p[i]<=crit[r-1]: thresh=p[i]
    sig=p<=thresh if thresh>0 else np.zeros_like(p,bool)
    return sig,thresh

# ---------- 5) Карти LISA + підсвітка цільових ТГ ----------
minx=min([s.bbox[0] for s in shapes]); miny=min([s.bbox[1] for s in shapes])
maxx=max([s.bbox[2] for s in shapes]); maxy=max([s.bbox[3] for s in shapes])
def draw_lisa(col, W, out_png, highlight_names=("Поляницька сільська громада","Яремчанська міська громада")):
    Ii,pvals,z,lag = local_moran(grid[col].values, W, 999, 2025)
    sig,thr = fdr_bh(pvals, 0.05)
    quad = np.full(len(z),"NS",object)
    quad[(z>0)&(lag>0)&sig]="HH"; quad[(z<0)&(lag<0)&sig]="LL"
    quad[(z>0)&(lag<0)&sig]="HL"; quad[(z<0)&(lag>0)&sig]="LH"
    cat_order=["NS","LL","LH","HL","HH"]; code=np.array([cat_order.index(q) for q in quad],float)
    Ig,pg = moran_global(grid[col].values, W, 999, 2025)
    fig=plt.figure(figsize=(6,7),dpi=300); ax=fig.add_subplot(111)
    coll=PatchCollection(patches, linewidths=0.35, edgecolor="black"); coll.set_array(code)
    ax.add_collection(coll); ax.set_xlim(minx,maxx); ax.set_ylim(miny,maxy)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"LISA {col} (kNN=5 by default); I={Ig:.3f}, p={pg:.3f}, FDR={thr:.3f}")
    # Підсвітка
    for nm in highlight_names:
        if nm in names:
            idx=names.index(nm); shp=shapes[idx]; pts=shp.points; parts=list(shp.parts)+[len(pts)]
            ring=pts[parts[0]:parts[1]] if len(parts)>=2 else pts
            ax.add_patch(Polygon(ring, closed=True, fill=False, linewidth=2.5))
            cx=np.mean([p[0] for p in ring]); cy=np.mean([p[1] for p in ring])
            ax.plot(cx,cy,marker="o",markersize=3); ax.text(cx,cy,nm.split()[0],fontsize=6,ha="left",va="bottom")
    cbar=fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.02); cbar.set_label("0=NS … 4=HH")
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

out_dir = BASE
for col in ["TN_km2","W_km2","AIR_km2","DW_km2","MSW_km2"]:
    draw_lisa(col, W5, str(out_dir/ f"lisa_{col}_knn5_highlight.png"))

# Макети А4 (2 панелі)
def assemble_a4(images, out_png, title):
    Wpx,Hpx=2480,3508; MARGIN=80; GAP=60
    page = Image.new("RGB",(Wpx,Hpx),"white"); draw=ImageDraw.Draw(page)
    try: font=ImageFont.truetype("DejaVuSans.ttf",50)
    except: font=ImageFont.load_default()
    draw.text((MARGIN,MARGIN), title, font=font, fill="black"); y=MARGIN+90
    cell_h=(Hpx-y-MARGIN-GAP)//2; cell_w=Wpx-2*MARGIN
    for i,pth in enumerate(images):
        im=Image.open(pth).convert("RGB")
        ir=im.width/im.height; cr=cell_w/cell_h
        if ir>cr: new_w=cell_w; new_h=int(cell_w/ir)
        else: new_h=cell_h; new_w=int(cell_h/ir)
        im=im.resize((new_w,new_h), Image.Resampling.LANCZOS)
        x=MARGIN+(cell_w-new_w)//2; y_i=y+i*(cell_h+GAP)+(cell_h-new_h)//2
        page.paste(im,(x,y_i))
    page.save(out_png); page.save(out_png.replace(".png",".pdf"))

assemble_a4([str(out_dir/"lisa_TN_km2_knn5_highlight.png"),
             str(out_dir/"lisa_W_km2_knn5_highlight.png")],
            str(out_dir/"A4_LISA_TN_W_highlight.png"),
            "LISA (kNN=5): TN/км² та W/км² (підсвічено Поляницьку й Яремчанську)")
assemble_a4([str(out_dir/"lisa_AIR_km2_knn5_highlight.png"),
             str(out_dir/"lisa_DW_km2_knn5_highlight.png")],
            str(out_dir/"A4_LISA_AIR_DW_highlight.png"),
            "LISA: AIR/км² та DW/км² (підсвічено Поляницьку й Яремчанську)")
assemble_a4([str(out_dir/"lisa_MSW_km2_knn5_highlight.png")],
            str(out_dir/"A4_LISA_MSW_highlight.png"),
            "LISA: MSW/км² (підсвічено Поляницьку й Яремчанську)")

# ---------- 6) FE-OLS на панелі + Moran I залишків і робастність ----------
# АЛІАСИ ваг, щоб збігалися імена змінних у нижньому блоці
W_knn4, W_knn5, W_queen = W4, W5, Wq

import numpy as np, pandas as pd, statsmodels.api as sm

# 0) Безпечне сортування панелі
panel_sorted = panel.sort_values(["year", "katottg"]).reset_index(drop=True)

# 1) Функція FE (within-OLS) — без дамі, без dtype=object
def fe_within_ols(y, X, entity, time):
    """
    Двосторонні фіксовані ефекти: демінування по ТГ (entity) і роках (time) + OLS (HC1).
    Повертає: dict(model=..., df=..., resid_d=деміновані залишки з ідентифікаторами e,t).
    """
    df = pd.concat([y.rename("y"), X.copy(),
                    entity.rename("e"), time.rename("t")], axis=1)

    # числові типи + чистка
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    for c in X.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"] + list(X.columns))

    # двостороннє демінування
    y_e = df.groupby("e")["y"].transform("mean")
    y_t = df.groupby("t")["y"].transform("mean")
    y_all = df["y"].mean()
    y_d = df["y"] - y_e - y_t + y_all

    X_d = pd.DataFrame(index=df.index)
    for c in X.columns:
        x_e = df.groupby("e")[c].transform("mean")
        x_t = df.groupby("t")[c].transform("mean")
        x_all = df[c].mean()
        X_d[c] = df[c] - x_e - x_t + x_all

    X_d = sm.add_constant(X_d.astype(float), has_constant="add")
    y_arr = y_d.values.astype(float)
    X_arr = X_d.values.astype(float)

    model = sm.OLS(y_arr, X_arr).fit(cov_type="HC1")

    resid_d = y_arr - model.predict(X_arr)
    out = {"model": model, "df": df[["e", "t"]].copy()}
    out["df"]["resid_d"] = resid_d
    return out

# 2) Оцінювання для кожного Y
targets_Y = ["AIR_km2", "DW_km2", "MSW_km2"]   # контролі додасте у X за потреби
results = {}
for Y in targets_Y:
    y = panel_sorted[Y]
    X = panel_sorted[["TN_km2"]]
    entity = panel_sorted["katottg"].astype(str)
    time = panel_sorted["year"].astype(int)
    try:
        results[Y] = fe_within_ols(y, X, entity, time)
        print(f"[FE-within] {Y}: OK")
    except Exception as e:
        print(f"[FE-within] {Y}: ERROR -> {e}")
        results[Y] = None

# 3) Зведення коефіцієнтів TN_km2 + AIC/BIC
coef_rows = []
for Y, res in results.items():
    if res is None:
        coef_rows.append({"Y": Y, "beta_TN_km2": np.nan, "SE_HC1": np.nan,
                          "p_value": np.nan, "AIC": np.nan, "BIC": np.nan})
        continue
    m = res["model"]
    # ім'я змінної TN_km2 завжди є серед exog_names (після константи)
    idx = m.model.exog_names.index("TN_km2") if "TN_km2" in m.model.exog_names else None
    beta = float(m.params[idx]) if idx is not None else np.nan
    se   = float(m.bse[idx])    if idx is not None else np.nan
    pv   = float(m.pvalues[idx])if idx is not None else np.nan
    coef_rows.append({"Y": Y, "beta_TN_km2": beta, "SE_HC1": se,
                      "p_value": pv, "AIC": float(m.aic), "BIC": float(m.bic)})
coef_df = pd.DataFrame(coef_rows).round(6)
coef_df.to_csv("fe_ols_TN_effects_2019_2024.csv", index=False)
print("[SAVE] fe_ols_TN_effects_2019_2024.csv")

# 4) Moran’s I для залишків (по роках, для різних W)
def moran_I_vec(r, W):
    r = r - r.mean()
    S0 = W.sum()
    num = len(r) * (r @ (W @ r))
    den = (r * r).sum()
    return float(num / (S0 * den)) if den > 0 else np.nan

resid_rows = []
for Y, res in results.items():
    if res is None: 
        continue
    df_res = res["df"][["e", "t", "resid_d"]].copy()
    for W_name, W in [("knn5", W_knn5), ("knn4", W_knn4), ("queen", W_queen)]:
        Is = []
        for yr in YEAR_RANGE:
            v = df_res.loc[df_res["t"] == yr, "resid_d"].values
            if len(v) == W.shape[0]:         # очікуємо рівно n ТГ на рік
                Is.append(moran_I_vec(v.astype(float), W))
        if len(Is) > 0:
            resid_rows.append({
                "Y": Y, "W": W_name,
                "I_resid_mean": float(np.nanmean(Is)),
                "I_resid_min": float(np.nanmin(Is)),
                "I_resid_max": float(np.nanmax(Is))
            })
resid_I_df = pd.DataFrame(resid_rows).round(6)
resid_I_df.to_csv("residuals_moranI_by_year.csv", index=False)
print("[SAVE] residuals_moranI_by_year.csv")

# 5) Робастність: вилучення «важковаговиків»
heavy = ["Ямницька сільська громада", "Калуська міська громада"]
mask = ~panel_sorted["name"].isin(heavy)

rob_rows = []
for Y in targets_Y:
    res_full = results.get(Y, None)
    m_full = res_full["model"] if res_full is not None else None

    try:
        res_rm = fe_within_ols(
            panel_sorted.loc[mask, Y],
            panel_sorted.loc[mask, ["TN_km2"]],
            panel_sorted.loc[mask, "katottg"].astype(str),
            panel_sorted.loc[mask, "year"].astype(int)
        )
        m_rm = res_rm["model"]
    except Exception as e:
        print(f"[ROBUST] {Y}: ERROR -> {e}")
        m_rm = None

    def get_b(model, name):
        if model is None or name not in model.model.exog_names: return np.nan
        idx = model.model.exog_names.index(name); return float(model.params[idx])
    def get_p(model, name):
        if model is None or name not in model.model.exog_names: return np.nan
        idx = model.model.exog_names.index(name); return float(model.pvalues[idx])

    rob_rows.append({
        "Y": Y,
        "beta_full": get_b(m_full, "TN_km2"), "p_full": get_p(m_full, "TN_km2"),
        "AIC_full": (float(m_full.aic) if m_full is not None else np.nan),
        "beta_rm": get_b(m_rm, "TN_km2"), "p_rm": get_p(m_rm, "TN_km2"),
        "AIC_rm": (float(m_rm.aic) if m_rm is not None else np.nan)
    })

rob_df = pd.DataFrame(rob_rows).round(6)
rob_df.to_csv("robustness_drop_heavyhitters.csv", index=False)
print("[SAVE] robustness_drop_heavyhitters.csv")

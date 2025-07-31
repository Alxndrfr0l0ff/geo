import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from esda.getisord import G_Local
from libpysal.weights import Queen
from esda.moran import Moran_BV

df_atm = pd.read_excel('revenue_pol_tax_atm.xlsx')
df_wtr = pd.read_excel('revenue_pol_tax_wtr.xlsx')
df_tpv = pd.read_excel('revenue_pol_tax_tpv.xlsx')
df_tour = pd.read_excel('revenue_tour_tax.xlsx')

print(df_atm.columns)
print(df_wtr.columns)
print(df_tpv.columns)

# Перейменовуємо колонки для уніфікації
df_atm = df_atm.rename(columns={'UPL': 'Податок_атмосфера'})
df_wtr = df_wtr.rename(columns={'UPL': 'Податок_вода'})
df_tpv = df_tpv.rename(columns={'UPL': 'Податок_ТПВ'})
df_tour = df_tour.rename(columns={'UPL': 'Туристичний_збір'})

# Об'єднуємо за кодом громади
df = df_atm.merge(df_wtr[['ST_NEW', 'Податок_вода']], on='ST_NEW', how='outer') \
           .merge(df_tpv[['ST_NEW', 'Податок_ТПВ']], on='ST_NEW', how='outer')\
           .merge(df_tour[['ST_NEW', 'Туристичний_збір']], on='ST_NEW', how='outer')    
           
df.to_excel('merged.xlsx', index=False)     

gdf = gpd.read_file('IF_reg_TG_bou_7.shp')
gdf_merged = gdf.merge(df, left_on='katotth', right_on='ST_NEW')

variables = [
    ('Податок_атмосфера', 'Податок за атмосферне забруднення'),
    ('Податок_вода', 'Податок за забруднення води'),
    ('Податок_ТПВ', 'Податок за розміщення ТПВ'),
    ('Туристичний_збір', 'Туристичний збір')
]

fig, axs = plt.subplots(1, 4, figsize=(20, 7))
for ax, (col, title) in zip(axs, variables):
    gdf_merged.plot(
        column=col,
        cmap='OrRd',
        linewidth=0.3,
        edgecolor='black',
        legend=True,
        ax=ax,
        legend_kwds={'shrink': 0.6}
    )
    ax.set_title(title, fontsize=13)
    ax.axis('off')
plt.suptitle('Геопросторовий розподіл надходжень екологічного податку та туристичного збору (2024)', fontsize=17, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])


for col in ['Податок_атмосфера', 'Податок_вода', 'Податок_ТПВ', 'Туристичний_збір']:
    y = gdf_merged[col].fillna(0).astype(float).values
    w = Queen.from_dataframe(gdf_merged)
    w.transform = 'r'
    gi = G_Local(y, w)
    gdf_merged[f'GiZ_{col}'] = gi.Zs
    gdf_merged[f'p_value_{col}'] = gi.p_sim
    
    # Візуалізація
    fig, ax = plt.subplots(figsize=(8, 7))
    gdf_merged.plot(
        column=f'GiZ_{col}',
        cmap='coolwarm',
        edgecolor='grey',
        linewidth=0.2,
        legend=True,
        ax=ax
    )
    ax.set_title(f'Гарячі точки (Getis-Ord Gi*) — {col}', fontsize=14)
    ax.axis('off')
    plt.show()
    
   
y1 = gdf_merged['Туристичний_збір'].fillna(0)
y2 = gdf_merged['Податок_атмосфера'].fillna(0)
y3 = gdf_merged['Податок_вода'].fillna(0)
y4 = gdf_merged['Податок_ТПВ'].fillna(0)
w = Queen.from_dataframe(gdf_merged)
w.transform = 'r'
moran_bv = Moran_BV(y1, y2, w)
moran_bv1 = Moran_BV(y1, y3, w)
moran_bv2 = Moran_BV(y1, y4, w)
print(f"Bivariate Moran's I: {moran_bv.I}, p-value: {moran_bv.p_sim}")
print(f"Bivariate Moran's I: {moran_bv1.I}, p-value: {moran_bv1.p_sim}")
print(f"Bivariate Moran's I: {moran_bv2.I}, p-value: {moran_bv2.p_sim}")
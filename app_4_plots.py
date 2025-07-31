import pandas as pd
import geopandas as gpd
from spreg import ML_Lag
from libpysal.weights import Queen
import numpy as np
import matplotlib.pyplot as plt

tourism = pd.read_csv('згруповано_турзбір.csv')
# Переконаємось у назвах колонок
print(tourism.columns)

ecol_atm = pd.read_excel('ecol_d1.xlsx')
ecol_wtr = pd.read_excel('ecol_d2.xlsx')
ecol_tpw = pd.read_excel('ecol_d3.xlsx')

atm_agg = ecol_atm.groupby(['HKATOTTG', 'РІК'], as_index=False).agg({'ОБСЯГ_ВИКИДУ_Т': 'sum'})
atm_agg.rename(columns={'ОБСЯГ_ВИКИДУ_Т': 'Викиди_атмосфера_т'}, inplace=True)

wtr_agg = ecol_wtr.groupby(['HKATOTTG', 'РІК'], as_index=False).agg({'ОБСЯГ_ВИКИДУ_Т': 'sum'})
wtr_agg.rename(columns={'ОБСЯГ_ВИКИДУ_Т': 'Викиди_вода_т'}, inplace=True)

tpw_agg = ecol_tpw.groupby(['HKATOTTG', 'РІК'], as_index=False).agg({'ОБСЯГ_ВИКИДУ_Т': 'sum'})
tpw_agg.rename(columns={'ОБСЯГ_ВИКИДУ_Т': 'ТПВ_т'}, inplace=True)

# Злиття (по 'Код_громади', 'Рік')
df = tourism.merge(atm_agg, left_on=['Код_громади', 'Рік'], right_on=['HKATOTTG', 'РІК'], how='left') \
            .merge(wtr_agg, left_on=['Код_громади', 'Рік'], right_on=['HKATOTTG', 'РІК'], how='left', suffixes=('', '_wtr')) \
            .merge(tpw_agg, left_on=['Код_громади', 'Рік'], right_on=['HKATOTTG', 'РІК'], how='left', suffixes=('', '_tpw'))

print(df.head())

gdf = gpd.read_file('IF_reg_TG_bou_7.shp')
print(gdf.columns)

gdf_merged = gdf.merge(df, left_on='katotth', right_on='Код_громади', how='left')
gdf_2024 = gdf_merged[gdf_merged['Рік'] == 2024]

# gdf_2024 - ваш GeoDataFrame для вибраного року (наприклад, 2024), вже містить усі потрібні колонки

variables = [
    ('Всього_туристо_діб', 'Туристо-доби'),
    ('Викиди_атмосфера_т', 'Викиди в атмосферу (т)'),
    ('Викиди_вода_т', 'Викиди у воду (т)'),
    ('ТПВ_т', 'ТПВ (т)')
]

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()

for ax, (col, title) in zip(axs, variables):
    gdf_2024.plot(
        column=col,
        cmap='OrRd',
        linewidth=0.3,
        edgecolor='black',
        legend=True,
        ax=ax,
        legend_kwds={'shrink': 0.7}
    )
    ax.set_title(title, fontsize=14)
    ax.axis('off')

plt.suptitle('Порівняння просторових патернів (2024 рік)', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

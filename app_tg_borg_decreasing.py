import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

gdf_boundaries = gpd.read_file("IF_reg_TG_bou_7.shp")
df_grouped = pd.read_excel("debt_decreasing.xlsx")

gdf_merged = gdf_boundaries.merge(df_grouped, left_on='katotth', right_on='ТГ')

top_10 = gdf_merged.nlargest(10, 'Погашено боргу, тис.грн.').copy()



# виправлена назва поля
top_10['short_name'] = top_10['Назва ТГ']

# будуємо карту
fig, ax = plt.subplots(figsize=(12, 12))
gdf_merged.plot(column='Погашено боргу, тис.грн.', cmap='OrRd', linewidth=0.3, edgecolor='black', legend=True, ax=ax)

ax.set_title("Інформація щодо погашення податкового боргу до бюджету ТГ з початку 2025 року", fontsize=16)
ax.axis('off')

# додаємо анотації для ТОП-10 громад
for idx, row in top_10.iterrows():
    plt.annotate(
        text=f"{row['short_name']}\n{row['Погашено боргу, тис.грн.']/1000:.1f} млн.",
        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=8,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
    )

# Додаємо таблицю ТОП-10 громад на карту
import pandas as pd
from matplotlib.table import Table

table_data = top_10[['short_name', 'Погашено боргу, тис.грн.']].copy()
table_data['Погашено боргу, тис.грн.'] = (table_data['Погашено боргу, тис.грн.']/1000).round(1).astype(str) 
table_data.columns = ['Назва ТГ', 'Погашено боргу, тис.грн.'.replace('Погашено боргу, тис.грн.', 'Погашено боргу, млн.грн.')]

# позиція таблиці
table_ax = fig.add_axes([0.05, 0.05, 0.25, 0.3])
table_ax.axis('off')
tbl = table_ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(table_data.columns))))
# компактний підпис над таблицею
table_ax.set_title("ТОП-10 територіальних громад\nза сумою погашеного боргу", fontsize=12, fontweight='bold')

plt.show()
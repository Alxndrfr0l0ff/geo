import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

gdf_boundaries = gpd.read_file("IF_reg_TG_bou_7.shp")
df_grouped = pd.read_excel("debt_b3060.xlsx")

gdf_merged = gdf_boundaries.merge(df_grouped, left_on='katotth', right_on='ТГ')

top_10 = gdf_merged.nlargest(10, 'Не опрацьовані').copy()



# виправлена назва поля
top_10['short_name'] = top_10['Назва ТГ']

# будуємо карту
fig, ax = plt.subplots(figsize=(12, 12))
gdf_merged.plot(column='Не опрацьовані', cmap='OrRd', linewidth=0.3, edgecolor='black', legend=True, ax=ax)

ax.set_title("Розподіл кількості боржників із сумою боргу більше 3060 грн., суми боргу яких не опрацьовані, в розрізі ТГ", fontsize=16)
ax.axis('off')

# додаємо анотації для ТОП-10 громад
for idx, row in top_10.iterrows():
    plt.annotate(
        text=f"{row['short_name']}\n{row['Не опрацьовані']:.0f} боржників \n{row['Сума боргу не опрацьованих']:.1f} млн.грн.",
        xy=(row.geometry.centroid.x, row.geometry.centroid.y),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=7,
        fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
    )

# Додаємо таблицю ТОП-10 громад на карту
import pandas as pd
from matplotlib.table import Table

table_data = top_10[['short_name', 'Не опрацьовані', 'Сума боргу не опрацьованих']].copy()
table_data['Не опрацьовані'] = (table_data['Не опрацьовані']).round(1).astype(str) 
table_data.columns = ['Назва ТГ', 'Кількість не опрацьованих', 'Сума боргу не опрацьованих, млн.грн.']

# позиція таблиці
table_ax = fig.add_axes([0.05, 0.05, 0.3, 0.3])
table_ax.axis('off')
tbl = table_ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(table_data.columns))))
# компактний підпис над таблицею
table_ax.set_title("ТОП-10 територіальних громад\nіз найбільшою кількістю боржників борг яких не опрацьований", fontsize=12, fontweight='bold')

plt.show()
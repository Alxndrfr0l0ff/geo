import geopandas as gpd
import folium

# Завантажуємо GeoJSON
gdf = gpd.read_file('IF_reg_TG_boundary.geojson')

# Фільтруємо лише полігони
gdf_poly = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

# Створюємо карту (центр — середнє координат)
center = [gdf_poly.geometry.centroid.y.mean(), gdf_poly.geometry.centroid.x.mean()]
m = folium.Map(location=center, zoom_start=9)

# Додаємо полігони з підказками
for _, row in gdf_poly.iterrows():
    folium.GeoJson(
        row.geometry,
        tooltip=row['name:uk']  # заміни на свою колонку з назвою
    ).add_to(m)

# Зберігаємо карту у файл
m.save('tg_map.html')
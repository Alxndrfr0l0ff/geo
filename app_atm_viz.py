import pandas as pd

# Завантаження файлів
ecol_atm = pd.read_excel('ecol_d1.xlsx')
dov = pd.read_excel('d1_dov.xlsx')

# Додаємо назву забруднюючої речовини
df = ecol_atm.merge(dov, left_on='КОД_ЗАБРУДНЮЮЧОЇ_РЕЧОВИНИ', right_on='Код', how='left')

# Агрегуємо обсяг викиду по громаді, року, речовині
df_grouped = df.groupby(['HKATOTTG', 'РІК', 'Назва'], as_index=False)['ОБСЯГ_ВИКИДУ_Т'].sum()

# Загальна сума по області (по речовинах і роках)
atm_total_by_pollutant_year = df_grouped.groupby(['РІК', 'Назва'], as_index=False)['ОБСЯГ_ВИКИДУ_Т'].sum()
atm_total_by_pollutant_year = atm_total_by_pollutant_year.sort_values(['РІК', 'ОБСЯГ_ВИКИДУ_Т'], ascending=[True, False])

# ТОП-10 речовин за останній рік
last_year = atm_total_by_pollutant_year['РІК'].max()
top_pollutants = atm_total_by_pollutant_year[atm_total_by_pollutant_year['РІК']==last_year-1].nlargest(10, 'ОБСЯГ_ВИКИДУ_Т')

top_pollutants[['Назва', 'ОБСЯГ_ВИКИДУ_Т']]
print(atm_total_by_pollutant_year)
print(top_pollutants)

atm_total_by_pollutant_year.to_excel('atm_total_by_pollutant_year.xlsx', index=False)

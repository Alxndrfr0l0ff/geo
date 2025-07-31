import pandas as pd
import glob

# 1. Завантаження та об'єднання всіх декларацій
files = glob.glob('test\decl3_max_1091627_2023_*.xls')
dfs = [pd.read_excel(file) for file in files]
decl_df = pd.concat(dfs, ignore_index=True)

# 2. Групування по коду речовини
decl_summary = decl_df.groupby('КОД_ЗАБРУДНЮЮЧОЇ_РЕЧОВИНИ', as_index=False)['ОБСЯГ_ВИКИДУ_Т'].sum()

# 3. Ваш основний файл
ecol_atm = pd.read_excel('ecol_d1.xlsx')
ecol_summary = ecol_atm[ecol_atm['РІК'] == 2024].groupby('КОД_ЗАБРУДНЮЮЧОЇ_РЕЧОВИНИ', as_index=False)['ОБСЯГ_ВИКИДУ_Т'].sum()

# 4. Злиття для порівняння
merged = decl_summary.merge(ecol_summary, on='КОД_ЗАБРУДНЮЮЧОЇ_РЕЧОВИНИ', suffixes=('_decl', '_ecol'))

# 5. Додаємо відносну різницю
merged['diff_abs'] = (merged['ОБСЯГ_ВИКИДУ_Т_decl'] - merged['ОБСЯГ_ВИКИДУ_Т_ecol']).abs()
merged['diff_rel_%'] = 100 * merged['diff_abs'] / merged['ОБСЯГ_ВИКИДУ_Т_decl'].replace(0, 1)

# 6. Для перегляду ТОП-10 найбільших відхилень
print(merged.sort_values('diff_rel_%', ascending=False).head(10))
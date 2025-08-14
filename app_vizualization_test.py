import pandas as pd

df = pd.read_excel("ecol_d1.xlsx")
dov = pd.read_excel("d1_dov.xlsx")
df_ = df.merge(dov, left_on="КОД_ЗАБРУДНЮЮЧОЇ_РЕЧОВИНИ", right_on="Код")
print(df_.head)

df_final = df_[df_["Код"]=="243.4.001"].groupby("РІК")["ОБСЯГ_ВИКИДУ_Т"].sum().reset_index()
print(df_final)
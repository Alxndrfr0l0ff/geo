import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def draw_curly_brace(ax, x, y0, y1, width=40, lw=4, color='black'):
    """
    Малює фігурну дужку (curly brace) як у Word/PowerPoint.
    x      - x-координата дужки (праворуч від bars)
    y0, y1 - вертикальний діапазон (від низу до верху)
    width  - наскільки далеко дужка "вигинається"
    lw     - товщина лінії
    color  - колір
    """
    h = (y1 - y0)
    verts = [
        (x, y0),                        # нижня точка
        (x + width, y0 + h*0.15),       # контрольна для нижньої дуги
        (x + width, y0 + h*0.35),       # контрольна для нижньої дуги
        (x, y0 + h*0.5),                # середина
        (x + width, y0 + h*0.65),       # контрольна для верхньої дуги
        (x + width, y0 + h*0.85),       # контрольна для верхньої дуги
        (x, y1),                        # верхня точка
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4
    ]
    path = Path(verts, codes)
    patch = PathPatch(path, fill=False, lw=lw, color=color, capstyle='round')
    ax.add_patch(patch)

# --- Дані ---
df = pd.read_excel('temp.xlsx')
df = df.rename(columns={df.columns[0]: "категорія"})
df_total = df[df['категорія'].str.lower().str.contains('всього')].copy()
df_total['дата'] = pd.to_datetime(df_total['дата'])
df_total['різниця'] = df_total['сума боргу, грн.'].diff()

labels = df_total['дата'].dt.strftime('%Y-%m')
values = df_total['сума боргу, грн.']
diffs = df_total['різниця']

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(labels, values, color='royalblue')

# --- Підписи на барах ---
for i, bar in enumerate(bars):
    value = values.iloc[i]
    ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
            f"{value:.1f}", va='center', ha='center', color='white',
            fontsize=18, fontweight='bold', rotation=0)

# --- Підписи різниці ---
for i, bar in enumerate(bars):
    if i == 0:
        continue
    різниця = diffs.iloc[i]
    color = 'green' if різниця < 0 else 'red'
    ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
            f"{різниця:+.1f}", va='center', ha='left', color=color,
            fontsize=14, fontweight='bold')

ax.set_xlabel('Сума боргу, млн грн', fontsize=14)
ax.set_title('Сума боргу до Зведеного бюджету по місяцях', fontsize=16)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
for spine in ax.spines.values():
    spine.set_visible(False)

# --- ДУЖКА і підпис ---
y0 = bars[0].get_y() + bars[0].get_height()/2
y1 = bars[-1].get_y() + bars[-1].get_height()/2
x_brace = bars[0].get_width() + max(values)*0.07

# Малюємо елегантну фігурну дужку
#draw_curly_brace(ax, x_brace + 20, y0, y1, width=60, lw=4, color='black')

# Підпис приросту
total_growth = values.iloc[-1] - values.iloc[0]
growth_color = 'red' if total_growth > 0 else 'green'
growth_text = f"ПРИРІСТ\n з початку року:\n{total_growth:+.1f} млн грн"
ax.text(
    x_brace + 110, (y0 + y1)/2,
    growth_text,
    va='center', ha='left',
    fontsize=16, fontweight='bold', color=growth_color
)

plt.tight_layout()
plt.show()

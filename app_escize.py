# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.patches import ArrowStyle
import matplotlib.patches as patches
import numpy as np

def sensor(ax, x, y, size=0.5):
    ax.add_patch(Rectangle((x-size*0.25, y), size*0.5, size*0.6))
    ax.add_patch(Rectangle((x-size*0.4, y+size*0.6), size*0.8, size*0.15))
    ax.add_patch(Rectangle((x-size*0.3, y+size*0.75), size*0.6, size*0.12))

def pole(ax, x, y, h=3.0):
    ax.add_line(plt.Line2D([x,x], [y, y+h], linewidth=2))
    return y+h

def checkmark(ax, x, y, s="рекомендовано"):
    ax.text(x, y, "✓ "+s, fontsize=10, ha="left", va="center")

def crossmark(ax, x, y, s="уникати"):
    ax.text(x, y, "✗ "+s, fontsize=10, ha="left", va="center")

def building(ax, x, y, w, h, floors=4):
    ax.add_patch(Rectangle((x,y), w, h, fill=False, linewidth=2))
    fh = h/(floors+1)
    for i in range(floors):
        ax.add_patch(Rectangle((x+w*0.1, y+fh*(i+0.2)), w*0.25, fh*0.6, fill=False, linewidth=1))
        ax.add_patch(Rectangle((x+w*0.65, y+fh*(i+0.2)), w*0.25, fh*0.6, fill=False, linewidth=1))

def tree(ax, x, y, r=0.9):
    ax.add_line(plt.Line2D([x, x], [y, y+0.8], linewidth=2))
    ax.add_patch(Circle((x, y+1.4), r, fill=False, linewidth=2))

def car(ax, x, y, w=2.5, h=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fill=False, linewidth=2))
    ax.add_patch(Circle((x+w*0.25, y), 0.22, fill=False, linewidth=2))
    ax.add_patch(Circle((x+w*0.75, y), 0.22, fill=False, linewidth=2))

def factory(ax, x, y):
    ax.add_patch(Rectangle((x, y), 1.0, 0.8, fill=False, linewidth=2))
    ax.add_patch(Rectangle((x+1.1, y), 1.2, 0.6, fill=False, linewidth=2))
    ax.add_patch(Rectangle((x+0.4, y), 0.3, 1.8, fill=False, linewidth=2))
    ax.add_patch(Rectangle((x+1.6, y), 0.4, 1.4, fill=False, linewidth=2))

def house(ax, x, y, w=1.6, h=1.2):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2))
    ax.add_patch(patches.Polygon([[x-0.1, y+h],[x+w+0.1, y+h],[x+w-0.2, y+h+0.4],[x+0.2, y+h+0.4]], fill=False, linewidth=2))

def no_symbol(ax, x, y, r=0.7):
    circle = Circle((x, y), r, fill=False, linewidth=3)
    ax.add_patch(circle)
    ax.add_line(plt.Line2D([x-r*0.7, x+r*0.7],[y-r*0.7, y+r*0.7], linewidth=3))

def arrow_double(ax, x1, y1, x2, y2, text=None):
    style = ArrowStyle("<->", head_length=4, head_width=2)
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle=style, linewidth=1.5))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2+0.15, text, ha="center", va="bottom", fontsize=9)

def setup_ax(ax, title):
    ax.set_aspect('equal'); ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.axis('off'); ax.text(0.5, 7.5, title, fontsize=14, ha="left", va="top")

# 4) Street canyon vs Open area
fig, axes = plt.subplots(1,2, figsize=(11,4), dpi=200)
setup_ax(axes[0], "Вуличний каньйон")
building(axes[0], 0.5, 0.5, 3.0, 6.5); building(axes[0], 8.5, 0.5, 3.0, 6.5)
sensor(axes[0], 4.3, 3.0, size=0.7); axes[0].add_line(plt.Line2D([3.5, 4.3],[3.35,3.35], linewidth=2))
axes[0].text(1.0, 6.5, "вільний обдув", fontsize=9)
arrow_double(axes[0], 3.5, 2.0, 9.0, 2.0, "2–4 м до фасадів")
axes[0].text(0.6, 0.2, "≥ 1,2 м над тротуаром", fontsize=9, ha="left")
checkmark(axes[0], 0.6, 1.0, "рекомендовано")

setup_ax(axes[1], "Відкрита місцевість")
top = pole(axes[1], 4.0, 0.5, h=3.2); sensor(axes[1], 4.0, 2.5, size=0.7)
tree(axes[1], 9.0, 0.5, r=1.0); arrow_double(axes[1], 4.0, 1.0, 9.0, 1.0, "≥ 2× висота перешкоди")
checkmark(axes[1], 0.5, 1.0, "достатній виступ"); crossmark(axes[1], 8.0, 0.5, "не під кронами/навісами")
fig.tight_layout(); fig.savefig("assets/04_street_vs_open.png", bbox_inches="tight"); plt.close(fig)

# 5) Industrial background vs roadside
fig, axes = plt.subplots(1,2, figsize=(11,4), dpi=200)
setup_ax(axes[0], "Промисловий фон"); factory(axes[0], 6.0, 1.5)
top = pole(axes[0], 2.5, 0.5, h=3.0); sensor(axes[0], 2.5, 2.3, size=0.7)
checkmark(axes[0], 0.6, 1.0, "репрезентативна локація")
setup_ax(axes[1], "Придорожня зона"); car(axes[1], 2.0, 0.8, w=3.0, h=1.0)
sensor(axes[1], 6.0, 1.3, size=0.7); crossmark(axes[1], 0.6, 0.3, "не ближче 10–30 м до дороги")
fig.tight_layout(); fig.savefig("assets/05_industrial_vs_road.png", bbox_inches="tight"); plt.close(fig)

# 6) Co-location with reference
fig, axes = plt.subplots(1,2, figsize=(11,4), dpi=200)
setup_ax(axes[0], "Співрозміщення з референсом — правильно")
axes[0].add_patch(Rectangle((3.5, 1.0), 2.2, 1.8, fill=False, linewidth=2))  # reference cabinet
top = pole(axes[0], 2.5, 0.5, h=3.2); sensor(axes[0], 2.5, 2.3, size=0.7)
arrow_double(axes[0], 2.5, 2.0, 4.6, 2.0, "однакова висота забору (2–4 м)")
checkmark(axes[0], 0.6, 0.5, "спільна експозиція повітря")
setup_ax(axes[1], "Співрозміщення — неправильно"); house(axes[1], 3.0, 0.8, w=2.5, h=1.6)
sensor(axes[1], 6.2, 1.3, size=0.7); no_symbol(axes[1], 6.2, 1.6, r=1.1)
crossmark(axes[1], 0.6, 0.5, "не всередині/під навісом")
fig.tight_layout(); fig.savefig("assets/06_colocation.png", bbox_inches="tight"); plt.close(fig)

# 7) Avoid shielding
fig, axes = plt.subplots(1,2, figsize=(11,4), dpi=200)
setup_ax(axes[0], "Уникати екранування — правильно")
top = pole(axes[0], 2.5, 0.5, h=3.2); sensor(axes[0], 2.5, 2.3, size=0.7)
tree(axes[0], 6.5, 0.5, r=1.0); arrow_double(axes[0], 2.5, 1.0, 6.5, 1.0, "відступ ≥ 2× висота крони")
checkmark(axes[0], 0.6, 0.3, "вільна роза вітрів")
setup_ax(axes[1], "Уникати екранування — неправильно"); building(axes[1], 6.5, 0.5, 3.0, 5.5)
sensor(axes[1], 6.8, 2.3, size=0.7); no_symbol(axes[1], 6.9, 2.8, r=1.0)
crossmark(axes[1], 0.6, 1.0, "впритул до стін/під кронами")
fig.tight_layout(); fig.savefig("assets/07_avoid_shielding.png", bbox_inches="tight"); plt.close(fig)

# 8) Mounting & power
fig, ax = plt.subplots(1,1, figsize=(8,5), dpi=200)
ax.set_aspect('equal'); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
ax.text(0.5, 7.5, "Монтаж і живлення LCS", fontsize=14, ha="left", va="top")
top = pole(ax, 2.2, 0.8, h=3.2); sensor(ax, 2.2, 2.6, size=0.7)
# height arrow vertical
ax.annotate("2–4 м висоти", xy=(2.2, 3.7), xytext=(2.2, 1.2),
            arrowprops=dict(arrowstyle="<->"), ha="center", va="center", fontsize=9)
ax.text(0.6, 0.8, "кронштейн/щогла, відступ від фасаду 0,5–1,0 м", fontsize=9)
# power
ax.add_patch(Rectangle((6.0, 4.6), 1.6, 1.0, fill=False, linewidth=2)); ax.text(6.8, 5.15, "AC 220В", ha="center", va="center", fontsize=10)
ax.add_patch(Rectangle((8.2, 4.6), 1.6, 1.0, fill=False, linewidth=2)); ax.text(9.0, 5.15, "Сонце+АКБ", ha="center", va="center", fontsize=10)
ax.text(6.8, 4.2, "заземлення, захист", ha="center", fontsize=9); ax.text(10.0, 4.2, "автономія ≥48 год", ha="center", fontsize=9)
# comms
ax.add_patch(Rectangle((6.0, 2.6), 1.6, 1.0, fill=False, linewidth=2)); ax.text(6.8, 3.1, "LTE/4G", ha="center", va="center", fontsize=10)
ax.add_patch(Rectangle((8.2, 2.6), 1.6, 1.0, fill=False, linewidth=2)); ax.text(9.0, 3.1, "Wi‑Fi", ha="center", va="center", fontsize=10)
ax.text(7.9, 2.1, "зовнішня антена за потреби", ha="center", fontsize=9)
ax.text(0.6, 0.2, "уникати прямих джерел тепла/димів; фіксувати кабелі; \nперевірити доступ для сервісу й безпеку", fontsize=9)
fig.savefig("assets/08_mount_power.png", bbox_inches="tight"); plt.close(fig)

print("OK")

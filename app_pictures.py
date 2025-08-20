# Re-run creation of three PNG sketches for micrositing rules.
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

out_dir = "20082025"

def save_fig(fig, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

# 1) Height & setbacks
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.set_xlim(0, 100); ax1.set_ylim(0, 60)
ax1.plot([0,100],[20,20], linewidth=3)           # road
ax1.plot([50,50],[0,40], linewidth=2)            # intersection
ax1.plot([0,100],[25,25], linestyle="--", linewidth=1)  # sidewalk
b = patches.Rectangle((70,25), 20, 20, fill=False, linewidth=1.5)
ax1.add_patch(b)                                 # building
ax1.plot([30,30],[25,38.5], linewidth=2)         # pole
ax1.plot(30,38.5, marker="o", markersize=6)      # sensor
ax1.annotate("≥ 10–15 м від перехрестя", xy=(35,20), xytext=(55,10), arrowprops=dict(arrowstyle="->"))
ax1.annotate("≥ 5–10 м від проїжджої частини", xy=(30,25), xytext=(5,35), arrowprops=dict(arrowstyle="->"))
ax1.annotate("Висота сенсора 3–4 м", xy=(30,38.5), xytext=(5,50), arrowprops=dict(arrowstyle="->"))
ax1.text(5,56,"Ескіз 1. Висота та відступи від трафіку/перехресть", fontsize=10)
ax1.axis("off")
p1 = save_fig(fig1, "micrositing_rule_1_height_setbacks.png")

# 2) Distance from local sources
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.set_xlim(0, 100); ax2.set_ylim(0, 60)
b2 = patches.Rectangle((20,20), 25, 20, fill=False, linewidth=1.5)
ax2.add_patch(b2)                                # building
ax2.plot([27,27],[40,52], linewidth=2)           # stack
ax2.plot([27,27],[52,55], linewidth=4)           # plume stub
ax2.plot([70,70],[20,33.5], linewidth=2)         # pole
ax2.plot(70,33.5, marker="o", markersize=6)      # sensor
ax2.annotate("", xy=(67,22), xytext=(27,22), arrowprops=dict(arrowstyle="<->"))
ax2.text(40,18,"≥ 25–30 м від джерела", ha="center")
ax2.annotate("Переважний вітер", xy=(40,50), xytext=(70,50), arrowprops=dict(arrowstyle="->"))
ax2.text(5,56,"Ескіз 2. Віддаленість від локальних джерел (котельня/вентвикид)", fontsize=10)
ax2.axis("off")
p2 = save_fig(fig2, "micrositing_rule_2_local_sources.png")

# 3) Valley & inversion (cross-section)
fig3, ax3 = plt.subplots(figsize=(6,4))
ax3.set_xlim(0, 100); ax3.set_ylim(0, 60)
x = np.linspace(0,100,200)
y = 30 + 10*np.cos((x-50)/50*np.pi)
ax3.plot(x,y, linewidth=2)                        # valley profile
ax3.plot(50,20, marker="o", markersize=6)         # low site
ax3.text(52,20,"Сенсор (долина, фон)", va="center")
ax3.plot(80,35, marker="o", markersize=6)         # higher site
ax3.text(82,35,"Сенсор (вище по схилу)", va="center")
ax3.plot([0,100],[32,32], linestyle="--")         # inversion layer
ax3.text(2,34,"Інверсія (ніч/зима)", fontsize=9)
ax3.annotate("", xy=(60,22), xytext=(40,22), arrowprops=dict(arrowstyle="->"))
ax3.text(50,17,"Вітер уздовж долини", ha="center")
ax3.text(5,56,"Ескіз 3. Долина: фон у низині + контроль вище по схилу", fontsize=10)
ax3.axis("off")
p3 = save_fig(fig3, "micrositing_rule_3_valley_inversion.png")

[p1, p2, p3]

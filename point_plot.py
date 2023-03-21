import os
import csv
import numpy as np
import matplotlib.pyplot as plt

source_fps = 30
target_fps = 10
frame_interval = source_fps // target_fps
interval_of_frame = 1 / target_fps

csv_dir = "outputs"
fig_target_dir = "figures"
if not os.path.isdir(f"{csv_dir}/{fig_target_dir}"):
    os.makedirs(f"{csv_dir}/{fig_target_dir}")

for fname in os.listdir(csv_dir):
    if "csv" not in fname or str(target_fps) not in fname:
        continue

    with open(f"{csv_dir}/{fname}", "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            if i == 0:
                xpoint = list(map(float, line))

            elif i == 1:
                ypoint = list(map(float, line))

            elif i == 2:
                width = list(map(float, line))

            elif i == 3:
                height = list(map(float, line))

            elif i == 4:
                page = list(map(float, line))

    xxy_change = []
    xy_change = []
    change = []
    m_change = []
    h0 = height[0] // 2

    xtick = np.array(range(1, len(height) + 1)) * interval_of_frame

    for i in range(len(xpoint) - 1):
        c = abs(height[i + 1] - height[i]) / h0
        # if abs(c)>20:
        #     print("warning")
        xy_change.append(c)
        m = abs(max(xy_change) - min(xy_change))
        m_change.append(m)

    # fig1 = plt.figure()
    # plt.plot(xtick[1:], m_change, linestyle="dotted")
    # plt.axhline(y=0.2, linestyle="dashed")
    # plt.ylim([0, 1])
    # plt.grid(True)
    # plt.show()

    fig = plt.figure()
    plt.plot(xtick[1:], xy_change)
    plt.axhline(y=0.2, linestyle="dashed")
    plt.grid(True)
    plt.xlabel("time (s)")
    plt.ylabel("differential of bbox height (abs)")
    plt.ylim([0, 1])
    plt.title(f"{fname}")
    # plt.show()
    fig.savefig(f"{csv_dir}/{fig_target_dir}/{fname[:-4]}.png", dpi=150)

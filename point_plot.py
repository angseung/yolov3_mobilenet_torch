import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

name = [116, 217]

source_fps = 30
target_fps = 10
frame_interval = source_fps // target_fps
interval_of_frame = 1 / frame_interval

for n in name:
    with open("listfile" + str(n) + ".csv", "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            if i == 0:
                xpoint = list(map(float, line))
            elif i == 1:
                ypoint = list(map(float, line))
            if i == 2:
                width = list(map(float, line))
            elif i == 3:
                height = list(map(float, line))
            else:
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

    # plt.plot(xpoint,ypoint)
    # plt.show()

    plt.plot(xtick, m_change, linestyle="dotted")
    plt.plot(xtick, xy_change)
    plt.grid(True)
    plt.ylim([0, 1])
    plt.show()

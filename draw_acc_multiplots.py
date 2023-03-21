import os
import csv
import numpy as np
import matplotlib.pyplot as plt

video_dir = "data/falldown"
csv_dir = "outputs"
fig_target_dir = "figures"
if not os.path.isdir(f"{csv_dir}/{fig_target_dir}"):
    os.makedirs(f"{csv_dir}/{fig_target_dir}")

for fname in os.listdir(video_dir):
    fname = fname[:-4]

    for idx, fps in enumerate([5, 10]):
        source_fps = 30
        frame_interval = source_fps // fps
        interval_of_frame = 1 / fps

        with open(f"{csv_dir}/{fname}_{str(fps)}fps.csv", "r", encoding="utf-8") as f:
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

        if len(height) < 2:
            continue

        xxy_change = []
        xy_change = []
        change = []
        m_change = []
        h0 = height[0] // 2

        xtick = np.array(range(1, len(height) + 1)) * interval_of_frame

        for i in range(len(xpoint) - 1):
            c = abs(height[i + 1] - height[i]) / h0

            xy_change.append(c)
            m = abs(max(xy_change) - min(xy_change))
            m_change.append(m)

        if idx == 0:
            fig = plt.figure()
            plt.axhline(y=0.2, linestyle="dotted")

        plt.plot(xtick[1:], xy_change, label=f"{str(fps)}fps")

        if idx == 1:
            plt.xlabel("time (s)")
            plt.ylabel("differential of bbox height (abs)")
            plt.ylim([0, 1])
            plt.title(f"{fname}")
            plt.grid(True)
            plt.legend()
            # plt.show()
            fig.savefig(f"{csv_dir}/{fig_target_dir}/{fname[:-4]}.png", dpi=150)

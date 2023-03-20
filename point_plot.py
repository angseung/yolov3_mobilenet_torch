import csv
import pandas as pd
import matplotlib.pyplot as plt

name = [159, 275, 363, 462, 560]
for n in name:
    with open("listfile" + str(n) + ".csv", "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for i, line in enumerate(rdr):
            if i == 0:
                xpoint = list(map(float, line))
            elif i == 1:
                ypoint = list(map(float, line))
            else:
                page = list(map(float, line))

    xxy_change = []
    xy_change = []
    change = []
    m_change = []

    for i in range(len(xpoint) - 1):
        c = (ypoint[i + 1] - ypoint[i]) / (page[i + 1] - page[i])
        # if abs(c)>20:
        #     print("warning")
        xy_change.append(c)
        m = abs(max(xy_change) - min(xy_change))
        m_change.append(m)

    # plt.plot(xpoint,ypoint)
    # plt.show()

    plt.plot(m_change, linestyle="dotted")
    plt.plot(xy_change)
    plt.show()

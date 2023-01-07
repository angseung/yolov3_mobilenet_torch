from typing import List
import platform
import numpy as np
from utils.classes_map import class_labels

labels_list = []

if __name__ == "__main__":
    assert len(class_labels) == len(set(class_labels))

    for index, value in enumerate(class_labels):
        print(f"{index} {value}")

    if "Windows" in platform.platform():
        label_hist = np.load(
            "./data/yperv2/labels/train.cache", allow_pickle=True
        ).item()
    elif "Linux" in platform.platform():
        label_hist = np.load(
            "/data_yper/yperv2/labels/train.cache", allow_pickle=True
        ).item()

    for i, (key, val) in enumerate(label_hist.items()):
        if key == "hash":
            break
        labels = val[0][:, 0].astype(np.uint8).flatten().tolist()
        labels_list += labels

    labels_list = np.array(labels_list)

    for index, label in enumerate(class_labels):
        bins = np.count_nonzero(labels_list == index)
        print(f"{index} {label} {bins}")

    print("label check done")

import shutil
import os

base_addr = "./data/yperv1/images"
base_addr_label = "./data/yperv1/labels"
with open("drop_list.txt", "r", encoding="utf-8") as f:
    img_list = f.readlines()

for line in img_list:
    fname = line.split()[0]

    if os.path.isfile(f"{base_addr}/train/{fname}.jpg"):
        shutil.move(
            f"{base_addr}/train/{fname}.jpg", f"{base_addr}/deleted/{fname}.jpg"
        )
        shutil.move(
            f"{base_addr_label}/train/{fname}.txt", f"{base_addr}/deleted/{fname}.txt"
        )

    if os.path.isfile(f"{base_addr}/val/{fname}.jpg"):
        shutil.move(f"{base_addr}/val/{fname}.jpg", f"{base_addr}/deleted/{fname}.jpg")
        shutil.move(
            f"{base_addr_label}/val/{fname}.txt", f"{base_addr}/deleted/{fname}.txt"
        )

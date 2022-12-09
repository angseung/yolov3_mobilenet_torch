import os
import shutil

base_dir = "./data/yperv1/images/train"
im_list_swim = os.listdir(base_dir)

with open("total_image_list.txt", "r") as f:
    im_list_sjan = f.readlines()

for fname in im_list_sjan:
    if im_list_swim.count(fname[:-1]) == 0:
        print(f"fname {fname} is not found.")

with open("delete_file_list.txt", "r") as f:
    delete_file_list = f.readlines()

for fname in delete_file_list:
    fname = fname[:-1]
    assert fname[-4:] == ".jpg"
    shutil.move(f"{base_dir}/{fname}", f"./data/yperv1/images/deleted/{fname}")

for fname in im_list_swim:
    if not os.path.isfile(f"./data/yperv1/labels/train/{fname[:-4]}.txt"):
        print(fname)

with open("validation_list.txt", "r") as f:
    validation_list = f.readlines()

for fname in validation_list:
    fname = fname[:-5]

    shutil.copy(
        f"./data/yperv1/images/train/{fname}.jpg",
        f"./data/yperv1/images/val/{fname}.jpg",
    )
    shutil.copy(
        f"./data/yperv1/labels/train/{fname}.txt",
        f"./data/yperv1/labels/val/{fname}.txt",
    )

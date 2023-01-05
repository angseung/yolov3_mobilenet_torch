import os
import shutil

base_dir = "./data/yperv1/images/train"
base_dir_labels = "./data/yperv1/labels/train"
im_list_swim = os.listdir(base_dir)
im_list_swim_labels = os.listdir(base_dir_labels)

with open("img_list.txt", "w") as f:
    for fname in im_list_swim:
        f.write(f"{fname[:-4]}\n")

# with open("total_image_list.txt", "r") as f:
#     im_list_sjan = f.readlines()

# for fname in im_list_sjan:
#     if im_list_swim.count(fname[:-1]) == 0:
#         print(f"fname {fname} is not found.")
#
# with open("delete_file_list.txt", "r") as f:
#     delete_file_list = f.readlines()
#
# for fname in delete_file_list:
#     fname = fname[:-1]
#     assert fname[-4:] == ".jpg"
#     shutil.move(f"{base_dir}/{fname}", f"./data/yperv1/images/deleted/{fname}")
#
# for fname in im_list_swim:
#     if not os.path.isfile(f"./data/yperv1/labels/train/{fname[:-4]}.txt"):
#         print(fname)
#
# for fname in im_list_swim_labels:
#     if not os.path.isfile(f"./data/yperv1/images/train/{fname[:-4]}.jpg"):
#         print(fname)

# with open("drop_list.txt", encoding="utf-8") as f:
#     f_list = f.readlines()
#
# for fname in f_list:
#     fname = fname[:-1]
#
#     if not os.path.isdir("/data_yper/yperv1/images/deleted"):
#         os.makedirs("/data_yper/yperv1/images/deleted")
#
#     shutil.move(
#         f"/data_yper/yperv1/images/train/{fname}.jpg",
#         f"/data_yper/yperv1/images/deleted/{fname}.jpg",
#     )
#     shutil.move(
#         f"/data_yper/yperv1/labels/train/{fname}.txt",
#         f"/data_yper/yperv1/labels/deleted/{fname}.txt",
#     )
#
# with open("validation_list.txt", "r") as f:
#     validation_list = f.readlines()
#
# for fname in validation_list:
#     fname = fname[:-5]
#
#     shutil.copy(
#         f"./data/yperv1/images/train/{fname}.jpg",
#         f"./data/yperv1/images/val/{fname}.jpg",
#     )
#     shutil.copy(
#         f"./data/yperv1/labels/train/{fname}.txt",
#         f"./data/yperv1/labels/val/{fname}.txt",
#     )

# 이미지 한장씩 변수로 받음
# fname 이미지 이름
import json
import os
from chitra.image import Chitra
import torch
from torchvision.transforms import ToTensor, ToPILImage
import traceback
from tqdm import tqdm

import cv2
import numpy as np
import os
import xmltodict
import json
from PIL import Image, ImageDraw
import sys, getopt


def drawBox(boxes, image, fname, image_save_path):

    for i in boxes:
        draw = ImageDraw.Draw(image)
        # changed color and width to make it visible
        draw.rectangle(tuple(i), outline=(0, 0, 255), width=3)
        image.save(os.path.join(image_save_path, fname))


def change_bbox_json(label, box_list, save_path, fname, x, y):

    label["@width"] = str(x)
    label["@height"] = str(y)

    for box_, change_box_ in zip(label["box"], box_list):

        box_["@xtl"] = str(change_box_[0])
        box_["@ytl"] = str(change_box_[1])
        box_["@xbr"] = str(change_box_[2])
        box_["@ybr"] = str(change_box_[3])

    ## json파일로 저장
    with open(os.path.join(save_path, fname[:-3]) + "json", "w") as f:
        json.dump(label, f, indent=4)


def resize_image_json(x, y, json_dir, json_save_path, image_save_path):

    # annotation_error_dic={}

    for json_file in tqdm(os.listdir(json_dir)):

        # 숨긴파일 삭제
        if json_file[0] == ".":
            continue

        # 저장된 json파일은 넘어가기
        # if json_file in check_list:
        #     continue

        with open(os.path.join(json_dir, json_file), "r") as file:
            dic = json.load(file)

        for label in dic["annotations"]["image"]:

            # label 이미지하나
            fname = label["@name"]
            # print(fname)

            # if fname in check_list:
            #     continue

            box_list = []
            try:

                for box_ in label["box"]:

                    if box_["@label"] != "car_plate":
                        continue

                    xtl = box_["@xtl"]
                    ytl = box_["@ytl"]
                    xbr = box_["@xbr"]
                    ybr = box_["@ybr"]

                    try:
                        number = box_["attribute"][0]["#text"]
                    except:

                        number = box_["attribute"][1]["#text"]

                    # print('label', number)

                    image = Chitra(
                        f"../LicensePlateRecognition_filtered_images/merge_images/{fname}",
                        (xtl, ytl, xbr, ybr),
                        number,
                    )

                    image, bboxes = image.resize_image_with_bbox((x, y))
                    coordinate = tuple(bboxes[0][0]) + tuple(bboxes[0][1])
                    # print('bbox', list(coordinate))

                    box_list.append(list(coordinate))

                # box plot하고 resize하여 저장
                drawBox(box_list, image, fname, image_save_path)

                # json 값 변경해서 저장
                # 이미지 하나씩 정제

                change_bbox_json(label, box_list, json_save_path, fname, x, y)

            except:
                continue

            # check wrong annotation


"""
                
                ("--------------------------------------")
                print("Invalid JSON file")
                print("json file", json_file)
                print("image_file", fname)
                print("                                   ")
                print("error message")
                
                print(traceback.format_exc())
                print("-------------------------------------------")
                
                
                try :
                    annotation_error_dic[json_file].append(fname)
                    
                    
                except:
                    annotation_error_dic[json_file]=[]
                    annotation_error_dic[json_file].append(fname)
   
            
                
"""


if __name__ == "__main__":

    x = 600
    y = 600
    # json file
    json_dir = ""
    json_save_path = "../data/yper_data/bbox_json"
    image_save_path = "../data/yper_data/bbox_image"

    resize_image_json(x, y, json_dir, json_save_path, image_save_path)

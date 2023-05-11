from PIL import ImageFont, ImageDraw, Image
import typing

def check_fall(func:str, ys: list[float], h0: float = 0):
    if func == "height":
        acc = (ys[1] - ys[0]) / h0
        if abs(acc) > 0.05:
            flag = 1
        else:
            flag = 0
            
    else:
        acc = ys[1] / ys[0]
        if abs(acc) > 1.05:
            flag = 1
        else:
            flag = 0
    return flag
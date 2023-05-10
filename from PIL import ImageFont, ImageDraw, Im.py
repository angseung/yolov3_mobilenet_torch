from PIL import ImageFont, ImageDraw, Image

def check_fall():
    for x_, y_ in zip(points_x, points_y):
                img_pillow = draw_point(img_pillow, (x_, y_))
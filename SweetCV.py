import cv2
import numpy as np
import math
import base64
from imageio import imread
import io
import ast

C_RATE = 0.0001

def contourApproximation(file):
    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(file)))

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_img, 100, 200)
    (_, cnts, _) = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    cnt = contours[0]
    perimeter = int(cv2.arcLength(cnt, True))
    epsilon = C_RATE * perimeter
    epsilon = epsilon / math.log(epsilon)
    if epsilon < 4:
        epsilon = 4
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    return approx.flatten()


def make_mask(width, height, point_json, mask_color):
    point = ast.literal_eval(point_json)

    img = np.zeros((height, width, 3), np.uint8)
    img2 = np.zeros((height, width, 3), np.uint8)

    mask_color = ast.literal_eval("[{}]".format(mask_color))

    point_root = np.array(point['root'], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [point_root], mask_color[:3])

    for i in range(len(point['add'])):
        point_add = np.array((point['add'][str(i)]), np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [point_add], mask_color[:3])

    for i in range(len(point['del'])):
        point_del = np.array((point['del'][str(i)]), np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img2, [point_del], mask_color[:3])

    img = cv2.subtract(img, img2)

    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * mask_color[
        3]  # creating a dummy alpha channel image.
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    retval, buffer = cv2.imencode('.png', img_BGRA)
    result = base64.b64encode(buffer)

    # img = imread(io.BytesIO(base64.b64decode(result)))
    # cv2.imwrite("test" + ".png", img)
    return result


if __name__=="__main__":
    make_mask(820,430,"{'add': {}, 'object_no': '1234', 'root': '[169,188,87,191,58,212,57,370,71,390,170,390,183,371,195,367,201,335,213,325,201,327,195,218]', 'del': {}, 'name': 'test'}","150,200,56,100")


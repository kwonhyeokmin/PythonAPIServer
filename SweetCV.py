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
    make_mask(720,960,'{"add":{"0":[186.666650390625,74.41666259765624,557.466650390625,112.81666259765625,465.066650390625,55.216662597656246],"1":[639.066650390625,667.2166625976562,243.066650390625,518.4166625976562,378.66665039062497,854.4166625976562,693.066650390625,927.6166625976562]},"object_no":13,"root":[576.666650390625,142.81666259765623,137.466650390625,97.21666259765625,133.86665039062498,427.21666259765624,159.066650390625,836.4166625976562,574.2666503906249,852.0166625976562,597.066650390625,508.8166625976562],"name":"person","del":{"0":[69.066650390625,199.21666259765624,663.066650390625,326.4166625976562,664.2666503906249,367.21666259765624,52.266650390624996,238.81666259765623]}}',"150,200,56,100")


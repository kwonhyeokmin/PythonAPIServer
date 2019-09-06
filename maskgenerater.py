import json
import cv2
import numpy as np
import ast
import base64

def make_mask(width, height, point_json, mask_color):
    point = ast.literal_eval(point_json)

    img = np.zeros((height, width, 3), np.uint8)
    img2 = np.zeros((height, width, 3), np.uint8)

    mask_color = ast.literal_eval("[{}]".format(mask_color))

    point_root = np.array(point['root'], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [point_root], mask_color)

    for i in range(len(point['add'])):
        point_add = np.array((point['add'][str(i)]), np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [point_add], mask_color)

    for i in range(len(point['del'])):
        point_del = np.array((point['del'][str(i)]), np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img2, [point_del], mask_color)

    img = img - img2
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50  # creating a dummy alpha channel image.
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    retval, buffer = cv2.imencode('.png', img_BGRA)
    return base64.b64encode(buffer)

if __name__=="__main__":
    width = 664
    height = 996
    json = '{"add":{},"object_no":3,"root":[312.49499999999995,46.065000000000005,286.34999999999997,48.55500000000001,261.45,62.25000000000001,247.75499999999994,87.15,239.03999999999994,125.74499999999999,227.38541666666663,148.70833333333334,220.46875,170.3229166666667,220.46875,200.58333333333334,220.46875,229.11458333333334,241.21874999999997,245.54166666666669,276.66666666666663,250.72916666666669,301.7395833333333,234.30208333333337,314.7083333333333,242.94791666666669,329.40625,246.40625000000006,347.5625,242.08333333333337,372.63541666666663,224.7916666666667,381.28124999999994,211.82291666666669,383.0104166666667,229.11458333333334,406.3541666666667,210.09375000000003,439.20833333333326,197.98958333333334,440.07291666666663,175.51041666666669,430.5625,135.73958333333334,407.21875,107.20833333333334,395.97916666666663,79.54166666666667,372.63541666666663,52.73958333333334,338.0520833333333,31.98958333333334,324.21875,33.71875000000001],"name":"head","del":{}}'
    color = "255,255,0,100"
    print(make_mask(width, height, json, color))
from flask import Flask, request
from SweetCV import contourApproximation, make_mask

app = Flask(__name__)


@app.route('/points', methods=['POST'])
def points():
    try:
        if request.method == 'POST':
            data = request.get_json()
            object_no = data["object_no"]
            object_nm = data["object_nm"]
            base64_img = data["base64_img"]
            contour = contourApproximation(base64_img)
            root_contour = "["
            for c in contour:
                root_contour += str(c)
                root_contour += ","
            root_contour = root_contour[:-1]
            root_contour += "]"
            result = '{"add":{},"object_no":%d, "root":%s, "del":{}, "name":"%s"}' % (int(object_no), root_contour, object_nm)
            return result
    except:
        return ""


@app.route('/mask', methods=['POST'])
def mask():
    if request.method == 'POST':
        try:
            data = request.get_json()
            width = int(data["width"])
            height = int(data["height"])
            json = data["pointsJson"]
            color = data["mask_color"]
            result = make_mask(width, height, json, color)
            return result
        except:
            return ""


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='5000')

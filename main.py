import cv2
import os
import numpy as np

IMG_DIR = "/home/gsh/PythonProjects/PycharmProjects/Black_spot_detection/data_imgs"


def show(window_name: str, img):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gaussblur(img, kernel=(5, 5), sigma=0):
    dst = cv2.GaussianBlur(img, kernel, sigma)
    return dst


def bgr2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def threshold(img):
    # dst=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,801,1)
    ret, dst = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return dst


def morphological_operation(binary_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (75, 75))
    dst_open = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel=kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    dst_close = cv2.morphologyEx(dst_open, cv2.MORPH_CLOSE, kernel=kernel2)

    return dst_close


def re_img(img):
    img_r = 255 - img
    return img_r


def findcounters(img):
    contuors, heriachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contuors


def drawcounters(img, contours):
    dst = cv2.drawContours(img, contours, -1, (0, 0, 255), 20)
    return dst


def drawbox(img, counters):
    for conter in counters:
        x, y, w, h = cv2.boundingRect(conter)
        # 画出重心
        # mm = cv2.moments(conter)
        # cx = mm["m10"] / mm["m00"]
        # cy = mm["m01"] / mm["m00"]
        # cv2.circle(img, (np.int(cx), np.int(cy)), 40, (0, 0, 200), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 15)
    return img


def main(img_dir):
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path, 1)
        img_gray = bgr2gray(img)
        img_gaussblur = gaussblur(img_gray)
        img_thresh = threshold(img_gaussblur)
        img_morphological = morphological_operation(img_thresh)

        img_morphological_r=re_img(img_morphological)

        counters = findcounters(img_morphological_r)
        # print(counters)
        img_draw = drawcounters(img.copy(), counters)
        img_box = drawbox(img.copy(), counters)
        #
        img_x = np.concatenate([img_gray, img_gaussblur, img_thresh, img_morphological,img_morphological_r], axis=1)
        show(img_name, img_x)
        img_x = np.concatenate([img, img_draw, img_box], axis=1)
        show(img_name, img_x)


if __name__ == '__main__':
    main(IMG_DIR)

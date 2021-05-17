import numpy as np
from PIL import Image
import sys
import glob, os

np.set_printoptions(threshold=sys.maxsize)

map_path = "m2.png"
draw_path = "23.png"
img_m = Image.open(map_path)
img_d = Image.open(draw_path)


imgm_arr = np.array(img_m)
imgd_arr = np.array(img_d)
output = np.zeros(imgm_arr.shape).astype(np.uint8)

def inner(arr):
    for a in arr:
        if a < 250:
            return False
    return True

def inner2(arr):
    if arr[0] < 100 and arr[1] < 100 and arr[2] > 200:
        return True
    else:
        return False

def change(img_arr, f):
    global output
    for i in range(0, img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            if f:
                if inner(img_arr[i][j]):
                    output[i][j][0] = 1
                    # img_arr[i][j] = np.array([255, 0, 0])
            else:
                if inner2(img_arr[i][j]):
                    output[i][j][1] = 1
                    # img_arr[i][j] = np.array([255, 0, 0])

change(imgm_arr, 1)
change(imgd_arr, 0)
print(output)
img = Image.fromarray(output)
img.show()

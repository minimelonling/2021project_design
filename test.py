import numpy as np
from PIL import Image
import sys

np.set_printoptions(threshold=sys.maxsize)

img = "ex1.png" 


arr = Image.open(img)
arr = np.array(arr)
out = np.zeros(arr.shape).astype(np.uint8)

for r in range(0, len(arr)):
    for c in range(0, len(arr[r])):
        f = True
        for pix in arr[r][c]:
            if pix < 250:
                f = False
                break
        if f:
            out[r][c] = np.array([255, 255, 255])

oimg = Image.fromarray(out)
oimg.show()

def 

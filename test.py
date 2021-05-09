import numpy as np
from PIL import Image
import sys

np.set_printoptions(threshold=sys.maxsize)

img = "ex1.png" 
draw = "23.png"

class block:
    def __init__(self):
        self.left = []
        self.right = []
        self.y = []
        self.len = 0

    def append(self, l = 0, r = 0, h = 0):
        self.left.append(l)
        self.right.append(r)
        self.y.append(h)
        self.len += 1

    def changelast(self, c, n):
        if c == 'l':
            self.left[self.len - 1] = n
        elif c == 'r':
            self.right[self.len - 1] = n
        elif c == 'y':
            self.y[self.len - 1] = n

blocks = []
path = []

def block_edge(img):
    global blocks
    active = []
    for r in range(0, len(img)):
        flag = False
        start = 0
        bk = -1
        for c in range(0, len(img[0])):
            if r == 0:
                if not flag and img[r][c][0] < 250:
                    #x = c + 5 if c + 5 < len(img[0]) + 1 else len(img[0])
                    #for i in range(c, x):
                    #    if img[r][i][0] >= 250:
                    #        break
                    #    elif i == x - 1:
                        blocks.append(block())
                        blocks[len(blocks) - 1].append(c, 0, r)
                        flag = True
                elif flag and img[r][c][0] >= 250:
                    #x = c + 5 if c + 5 < len(img[0]) + 1 else len(img[0])
                    #for i in range(c, x):
                    #    if img[r][i][0] < 250:
                    #        break
                    #    elif i == x - 1:
                        blocks[len(blocks)- 1].changelast('r', c)
                        active.append(len(blocks) - 1)
                        flag = False
            else:
                if img[r][c][0] < 250:
                    if not flag:
                        start = c
                    flag = True
                    if bk == -1:
                        for a in active:
                            if blocks[a].left[blocks[a].len - 1] <= c and blocks[a].right[blocks[a].len - 1] >= c:
                                blocks[a].append(start, 0, r)
                                bk = a
                                break
                elif img[r][c][0] >= 250:
                    if flag:
                        if bk != -1:
                            blocks[bk].changelast('r', c)
                            bk = -1
                        else:
                            blocks.append(block())
                            blocks[len(blocks) - 1].append(start, c, r)
                            active.append(len(blocks) - 1)
                        flag = False
        i = 0
        l = len(active)
        while(i < l):
            if blocks[active[i]].y[blocks[active[i]].len - 1] != r:
                active.pop(i)
                l -= 1
            i += 1

    k = 0
    for b in blocks:
        print(k)
        for i in range(0, b.len):
            print(b.left[i], b.right[i], b.y[i])
        k += 1

"""
def find_path(img):
    step = 3
    for i in range(0, len(img), step):
        for j in range(0, len(img[0]), step):
            if img[i][j][0] < 100 and img[i][j][1] < 100 and img[i][j][2] > 200:
                path.append(list([i, j]))

def fit_map(w, h):
    for b in blocks:
        up = h
        down = 0
        left = w
        right = 0
        for p in path:
            for i in range(0, b.len):
                if b.y[i] == p[0] and b.left[i] < p[1] and b.right[i] > p[1]:
                    if b[0] < up:
                        up = b[0]
                    if b[0] > down:
                        down = b[0]
                    if b[1] < left:
                        left = b[1]
                    if b[1] > right:
                        right = b[1]
            def linear(u, d, l, r, x, y):
                if u == d:
                    return y - u
                elif l == r:
                    return x - l
                else:
                    midx = (l + r) / 2
                    midy = (u + d) / 2
                    m = (d - u) / 
                    return 
            sec = [0, 0]
            if 
"""

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

block_edge(out)
oimg = Image.fromarray(out)
oimg.save("1.png")

for b in blocks:
    for i in range(0, b.len):
        for k in range(0, 5):
            if b.left[i] + k < len(out[0]):
                out[b.y[i]][b.left[i] + k] = np.array([255, 0, 0])
            if b.right[i] + k < len(out[0]):
                out[b.y[i]][b.right[i] + k] = np.array([255, 0, 0])

oimg = Image.fromarray(out)
oimg.save("2.png")


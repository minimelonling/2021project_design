import numpy as np
from PIL import Image
import sys

np.set_printoptions(threshold=sys.maxsize)

class pair_dot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class block:
    def __init__(self, e):
        self.edge = e

blocks = []
def fit_map(m, p):
    global blocks
    t = 1
    for i in range(0, len(m)):
        print("i = ", i)
        for j in range(0, len(m[0])):
            #if m[i][j][0] < 250 and p[i][j][0] < 100 and p[i][j][1] < 100 and p[i][j][2] > 200:
            if m[i][j][0] < 250 and m[i][j][1] == 0:
                blocks.append(block(get_edge(m, j, i, t)))
                t += 1
                """
                for k in range(0, len(blocks)):
                    for e in blocks[k].edge:
                        if e.y == i and e.x[0] <= j and e.x[1] >= j:
                            k = len(blocks)
                            break
                    if k == len(blocks) - 1:
                        blocks.append(block(get_edge(m, j, i)))
                if len(blocks) == 0:
                    blocks.append(block(get_edge(m, j, i)))
                """

    for b in blocks:
        for e in b.edge:
            print(len(b.edge), len(blocks))
            k1 = e.x[0] + 2 if e.x[0] + 2 < len(m[0]) else len(m[0])
            k2 = e.x[1] + 2 if e.x[1] + 2 < len(m[0]) else len(m[0])
            for i in range(0, k1):
                m[e.y][e.x[0]] = [255, 0, 0]
            for i in range(0, k2):
                m[e.y][e.x[0]] = [255, 0, 0]
                """
                for e in edge:
                    for k in range(e.x[0], e.x[1]):
                        m[e.y][k] = [0, t, 255]
                t += 1        
                """
                    #for i in range(0, 5):
                        #m[e.y][e.x[0] + i if e.x[0] + i < len(m[0]) else e.x[0]] = [0, 0, 255]
                        #m[e.y][e.x[1] + i if e.x[1] + i < len(m[0]) else e.x[1]] = [0, 255, 255]

def get_edge(m, x, y, t):
    left = x
    right = x
    edge = []
    while left > 0 and m[y][left-1][0] < 250:
        m[y][left][1] = t 
        left -= 1
    while right < len(m[0]) - 1 and m[y][right+1][0] < 250:
        m[y][right][1] = t
        right += 1
    edge.append(pair_dot([left, right], y))
    tmp_pair = [left, right]
    tmp_y = y
    while tmp_y > 0:
        tmp_y -= 1
        find_new_pair(tmp_pair, tmp_y, m, t)
        if tmp_pair[0] > tmp_pair[1]:
            break
        edge.insert(0, pair_dot([tmp_pair[0], tmp_pair[1]], tmp_y))

    tmp_pair = [left, right]
    tmp_y = y
    while tmp_y < len(m) - 1:
        tmp_y += 1
        find_new_pair(tmp_pair, tmp_y, m, t)
        if tmp_pair[0] > tmp_pair[1]:
            break
        edge.append(pair_dot([tmp_pair[0], tmp_pair[1]], tmp_y))
    return edge

def find_new_pair(tmp_pair, tmp_y, m, t):
    if m[tmp_y][tmp_pair[0]][0] < 250:
        m[tmp_y][tmp_pair[0]][1] = t
        while tmp_pair[0] - 1 >= 0 and m[tmp_y][tmp_pair[0] - 1][0] < 250:
            m[tmp_y][tmp_pair[0] - 1][1] = t
            tmp_pair[0] -= 1
    else:
        while tmp_pair[0] < len(m[0]) and m[tmp_y][tmp_pair[0]][0] >= 250:
            tmp_pair[0] += 1
    if m[tmp_y][tmp_pair[1]][0] < 250:
        m[tmp_y][tmp_pair[1]][1] = t
        while tmp_pair[1] + 1 < len(m[0]) and m[tmp_y][tmp_pair[1] + 1][0] < 250:
            m[tmp_y][tmp_pair[0] + 1][1] = t
            tmp_pair[1] += 1
    else:
        while tmp_pair[1] >= 0 and m[tmp_y][tmp_pair[1]][0] >= 250:
            tmp_pair[1] -= 1


img = "ex1.png" 
img1 = "23.png"
arr1 = Image.open(img)
arr2 = np.array(Image.open(img1).resize(arr1.size))
arr1 = np.array(arr1)
out = np.zeros(arr1.shape).astype(np.uint8)

for r in range(0, len(arr1)):
    for c in range(0, len(arr1[r])):
        f = True
        for pix in arr1[r][c]:
            if pix < 250:
                f = False
                break
        if f:
            out[r][c] = np.array([255, 255, 255])

fit_map(out, arr2)
oimg = Image.fromarray(out)
oimg.save("out.png")
oimg = Image.fromarray(arr2)
oimg.save("in2.png")

import numpy as np
from PIL import Image
import sys
import glob, os
import copy

np.set_printoptions(threshold=sys.maxsize)

# map_path = "m2.png"
map_path = "ex1.png"
draw_path = "23.png"
img_m = Image.open(map_path)
imgm_arr = np.array(img_m)
img_d = Image.open(draw_path).resize(img_m.size)
imgd_arr = np.array(img_d)
output = np.zeros(imgm_arr.shape).astype(np.uint8)

class pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class block:
    def __init__(self, p, parent):
        self.point = copy.deepcopy(p)
        self.parent = parent

road = []
def get_road(m, f):
    global road
    if f == 0:
        for i in range(0, len(m)):
            bk = 0
            start = 0
            end = 0
            ms = 0
            for j in range(0, len(m[0])):
                if m[i][j][0] > 250 and not j == len(m[0]) - 1:
                    bk += 1
                    ms = 0
                else:
                    if ms > 3:
                        if bk > 3:
                            road.append(pos(int((start + end) / 2), i))
                            bk = 0
                        start = j
                    elif ms == 0:
                        end = j
                    ms += 1
    else:
        for i in range(0, len(m[0])):
            bk = 0
            start = 0
            end = 0
            ms = 0
            for j in range(0, len(m)):
                if m[j][i][0] > 250 and not j == len(m) - 1:
                    bk += 1
                    ms = 0
                else:
                    if ms > 3:
                        if bk > 3:
                            road.append(pos(i, int((start + end) / 2)))
                            bk = 0
                        start = j
                    elif ms == 0:
                        end = j
                    ms += 1

def print_road(m):
    global road
    for r in road:
        for i in range(0, 2):
            if r.x + i < len(m[0]):
                m[r.y][r.x + i] = [255, 0, 0]

def find_path():
    global output
    global draw_road
    detect_block = 20
    start = []
    for p in draw_road:
        up = p.y - detect_block if p.y - detect_block >= 0 else 0
        down = p.y + detect_block if p.y + detect_block < len(output) else len(output) - 1
        left = p.x - detect_block if p.x - detect_block >= 0 else 0
        right = p.x + detect_block if p.x + detect_block < len(output[0]) else len(output[0]) - 1
        in_block = []
        for i in range(up, down):
            for j in range(left, right):
                if output[i][j][0] == 255:
                    in_block.append(pos(j, i))
                    output[i][j][1] = 255
                    output[i][j][2] = 1
        if not len(in_block) == 0:
            start.append(block(in_block, None))

    root = [start.pop(0)]
    for p in root[0].point:
        output[p.y][p.x][2] = 2
    paths = []
    while(not len(start) == 0):
#    for i in range(0, 2):
        paths.append(link_pos(root, start, detect_block))
        print("start: ", len(start))

def link_pos(root, s, detect_block):
    global output
    max_iter = 10000000
    cur_iter = []
    next_iter = copy.deepcopy(root)
    ret = []
    #for time in range(0, max_iter):
    while True:
        cover = []
        cur_iter = copy.deepcopy(next_iter)
        next_iter = []
        for cur in cur_iter:
            l = len(output[0]) - 1
            r = 0
            u = len(output)
            d = 0
            for p in cur.point:
                if p.x < l:
                    l = p.x
                if p.x > r:
                    r = p.x
                if p.y < u:
                    u = p.y
                if p.y > d:
                    d = p.y

            left = 0
            right = 0
            up = 0
            down = 0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not (i == 0 and j == 0):
                        if l + detect_block * i < 0:
                            left = 0
                        elif l + detect_block * i >= len(output[0]):
                            left = len(output[0]) - 1
                        else:
                            left = l + detect_block * i
                    
                        if r + detect_block * i < 0:
                            right = 0
                        elif r + detect_block * i >= len(output[0]):
                            right = len(output[0]) - 1
                        else:
                            right = r + detect_block * i
                    
                        if u + detect_block * i < 0:
                            up = 0
                        elif u + detect_block * i >= len(output):
                            up = len(output) - 1
                        else:
                            up = u + detect_block * i
                    
                        if d + detect_block * i < 0:
                            down = 0
                        elif d + detect_block * i >= len(output):
                            down = len(output) - 1
                        else:
                            down = d + detect_block * i

                        in_block = []
                        for k in range(up, down):
                            for l in range(left, right):
                                if output[k][l][0] == 255:
                                    in_block.append(pos(l, k))
                                    if output[k][l][2] == 1:
                                        for node in s:
                                            f = True
                                            for c in cover:
                                                if c == node:
                                                    f = False
                                            if f:
                                                for p in node.point:
                                                    if p.x == l and p.y == k:
                                                        cover.append(node)
                                                        break
                        if len(in_block) > 0:
                            next_iter.append(block(in_block, cur))
        # debug
        for n in next_iter:
            for p in n.point:
                output[p.y][p.x][1] = 255

        if len(cover) > 0:
            print("pop")
            for node in cover:
                tmp = node
                for p in node.point:
                    output[p.y][p.x][2] = 2
                while not tmp == None:
                    ret.append(tmp)
                    tmp = tmp.parent
                i = 0
                while i != len(s):
                    if s[i] == node:
                        root.append(s.pop(i))
                    else:
                        i += 1
            break
                    
                                
                                
                            



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
            if f == 1:
                if not inner(img_arr[i][j]):
                    output[i][j] = [0, 0, 0]
                else:
                    output[i][j] = [255, 255, 255]
                    # img_arr[i][j] = np.array([255, 0, 0])
            else:
                if inner2(img_arr[i][j]):
                    output[i][j] = [255, 255, 255]
                    # img_arr[i][j] = np.array([255, 0, 0])

change(imgm_arr, 1)
get_road(output, 0)
get_road(output, 1)
map_road = copy.deepcopy(road)
print_road(output)
output1 = copy.deepcopy(output)
road = []
output = np.zeros(imgm_arr.shape).astype(np.uint8)
change(imgd_arr, 0)
get_road(output, 0)
get_road(output, 1)
draw_road = copy.deepcopy(road)
print_road(output)
output2 = copy.deepcopy(output)

output = np.zeros(imgm_arr.shape).astype(np.uint8)
for p in map_road:
    output[p.y][p.x][0] = 255
find_path()

#print(output)
img = Image.fromarray(output1)
img.save("t1.png")
img = Image.fromarray(output2)
img.save("t2.png")
img = Image.fromarray(output)
img.save("t3.png")

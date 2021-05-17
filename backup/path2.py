import numpy as np
from PIL import Image
import sys
import glob, os
import copy
import math

np.set_printoptions(threshold=sys.maxsize)

# map_path = "m2.png"
map_path = "ex1.png"
draw_path = "28.png"
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
    def __init__(self, up, down, left, right, parent):
        self.u = up
        self.d = down
        self.l = left
        self.r = right
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


    detect_block = 10
    start = []
    for p in draw_road:
        up = p.y - detect_block if p.y - detect_block >= 0 else 0
        down = p.y + detect_block if p.y + detect_block < len(output) else len(output) - 1
        left = p.x - detect_block if p.x - detect_block >= 0 else 0
        right = p.x + detect_block if p.x + detect_block < len(output[0]) else len(output[0]) - 1
        store = False
        for i in range(up, down):
            for j in range(left, right):
                if output[i][j][0] == 255:
                    store = True
                    output[i][j][1] = 255
                    output[i][j][2] = 1
        if store:
            tmp = block(up, down, left, right, None)
            start.append(tmp)

    root = [start.pop(0)]
    for i in range(root[0].u, root[0].d):
        for j in range(root[0].l, root[0].r):
            if output[i][j][2] == 1:
                output[i][j][2] == 2
    paths = []
    while(not len(start) == 0):
#    for i in range(0, 2):
        paths.append(link_pos(root, start, detect_block * 2))
        print("start: root: ", len(start), len(root))

def link_pos(root, s, detect_block):
    global output
    max_iter = 10000000
    cur_iter = []
    next_iter = copy.deepcopy(root)
    ret = []
    #for time in range(0, max_iter):
    while len(next_iter) < 10000:
        cover = []
        cur_iter = copy.deepcopy(next_iter)
        next_iter = []
        ins = 0
        for cur in cur_iter:
            left = 0
            right = 0
            up = 0
            down = 0

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not (i == 0 and j == 0):
                        if cur.l + detect_block * i < 0:
                            left = 0
                        elif cur.l + detect_block * i >= len(output[0]):
                            left = len(output[0]) - 1
                        else:
                            left = cur.l + detect_block * i
                    
                        if cur.r + detect_block * i < 0:
                            right = 0
                        elif cur.r + detect_block * i >= len(output[0]):
                            right = len(output[0]) - 1
                        else:
                            right = cur.r + detect_block * i
                    
                        if cur.u + detect_block * i < 0:
                            up = 0
                        elif cur.u + detect_block * i >= len(output):
                            up = len(output) - 1
                        else:
                            up = cur.u + detect_block * i
                    
                        if cur.d + detect_block * i < 0:
                            down = 0
                        elif cur.d + detect_block * i >= len(output):
                            down = len(output) - 1
                        else:
                            down = cur.d + detect_block * i
                        if up > down:
                            print("1", up <= down)
                        if left > right:
                            print("2", left <= right)
                        store = False
                        for k in range(up, down):
                            for l in range(left, right):
                                if output[k][l][0] == 255:
                                    store = True
                                    if output[k][l][2] == 1:
                                        for node in s:
                                            f = True
                                            for c in cover:
                                                if c == node:
                                                    f = False
                                            if f:
                                                if node.u <= k and node.d >= k:
                                                    if node.l <= l and node.r >= l:
                                                        cover.append(node)
                                                        break
                                    elif output[k][l][2]:
                                        store = False
                        ins += 1
                        if store:
                            #print("store", ins)
                            next_iter.append(block(up, down, left, right, cur))
        #debug
        for n in next_iter:
            for i in range(n.u, n.d):
                for j in range(n.l, n.r):
                    if output[i][j][0] == 255:
                        output[i][j][1] = 255

        if len(cover) > 0:
            print("pop")
            for node in cover:
                tmp = node
                for i in range(node.u, node.d):
                    for j in range(node.l, node.r):
                        if output[i][j][2] == 1:
                            output[i][j][2] = 2
                while not tmp == None:
                    ret.append(tmp)
                    tmp = tmp.parent
                i = 0
                while i != len(s):
                    if s[i] == node:
                        root.append(s.pop(i))
                    else:
                        i += 1
            return ret
                    
def get_sequence(detect_block):
    global output
    global draw_road
    """
    detect_block = 5
    blocks = []
    for d in draw_road;
        l = d.x - detect_block if d.x - detect_block >= 0 else 0
        r = d.x + detect_block if d.x + detect_block < len(output[0]) else len(output[0]) - 1
        u = d.y - detect_block if d.y - detect_block >= 0 else 0
        d = d.y + detect_block if d.y + detect_block < len(output) else len(output) - 1
        blocks.append(block(u, d, l, r, None))
    """
    #seq = [blocks.pop(0)]
    seq = [draw_road.pop(0)]
    a = 0
    it = 0
    max_iter = 10000
    while len(draw_road) > 0 and it < max_iter:
        it += 1
        print(len(draw_road))
        i = 0
        """
        output = np.zeros(imgm_arr.shape).astype(np.uint8)
        for s in seq:
            output[s.y][s.x] = [255, 0, 0]
        img = Image.fromarray(output)
        img.save("a" + str(a) + ".png")
        a += 1
        """
        while True:
            if i >= len(draw_road):
                break
            if len(seq) == 1:
                if distance(seq[0], draw_road[i]) < detect_block:
                    seq.append(draw_road.pop(i))
            else:
                dist = []
                for j in range(0, len(seq) - 1):
                    d1 = distance(seq[j], draw_road[i])
                    d2 = distance(seq[j + 1], draw_road[i])
                    if d1 < detect_block and d2 < detect_block:
                        dist.append((j + 1, d1 + d2))
                if len(dist) == 0:
                    i += 1
                    continue
                mn = 0
                for d in range(0, len(dist)):
                    if dist[d][1] < dist[mn][1]:
                        mn = d
                if dist[mn][0] == 1:
                    tmp = tri_test(seq[0], seq[1], draw_road[i])
                    seq.pop(0)
                    seq.pop(0)
                    for i in range(0, 3):
                        seq.insert(i, tmp[i])
                elif dist[mn][0] == len(dist):
                    tmp = tri_test(seq[len(seq) - 2], seq[len(seq) - 1], draw_road[i])
                    seq.pop(len(seq) - 1)
                    seq.pop(len(seq) - 1)
                    for i in range(0, 3):
                        seq.append(tmp[i])
                else:
                    seq.insert(dist[mn][0], draw_road[i])
                print(i, len(draw_road))
                if i >= len(draw_road):
                    break
                draw_road.pop(i) 
    return seq        

def fit_on_map(seq):
    global output1
    part = []
    detect_block = 2
    fit = []
    u = seq[0].y
    while u >= 0:
        if output1[u][seq[0].x][0] == 255:
            fit.append(pos(seq[0].x, u))
            break
        else:
            u -= 1
    if u == 0:
        u = seq[0].y
        while u < len(output1):
            if output1[u][seq[0].x][0] == 255:
                fit.append(pos(seq[0].x, u))
                break
            else:
                u += 1
    if u == len(output1):
        u = seq[0].x
        while u >= 0:
            if output1[seq[0].y][u][0] == 255:
                fit.append(pos(u, seq[0].y))
                break
            else:
                u -= 1
    if u == 0:
        u = seq[0].x
        while u < len(output1[0]):
            if output1[seq[0].y][u][0] == 255:
                fit.append(pos(u, seq[0].y))
                break
            else:
                u += 1

    if len(fit) == 0:
        fit.append(seq[0])
    for s in seq:
        blk = get_range(s, detect_block)
        flag = False
        for i in range(blk[0], blk[1]):
            for j in range(blk[2], blk[3]):
                if output1[i][j][0] == 255:
                    flag = True
        if not flag:
            part.append(s)
        else:
            fit_road(fit[len(fit) - 1], s, part, fit)
            fit.append(s)
            part = []
    return fit

def get_range(p, detect_block):
    up = p.y - detect_block if p.y - detect_block >= 0 else 0
    down = p.y + detect_block if p.y + detect_block < len(output) else len(output) - 1
    left = p.x - detect_block if p.x - detect_block >= 0 else 0
    right = p.x + detect_block if p.x + detect_block < len(output[0]) else len(output[0]) - 1
    return [up, down, left, right]

def fit_road(a, b, part, fit):
    global output1
    midx = int((a.x + b.x) / 2)
    midy = int((a.y + b.y) / 2)
    if abs(a.x - b.x) > abs(a.y - b.y):
        tmpy1 = midy
        tmpy2 = midy
        while tmpy1 >= 0 and output1[tmpy1][midx][0] < 250:
            tmpy1 -= 1
        while tmpy2 < len(output1) and output1[tmpy2][midx][0] < 250:
            tmpy2 += 1
        direction = 1 if tmpy1 > tmpy2 else -1

        if a.x > b.x:
            left = len(part) - 1
            right = 0
        else:
            left = 0
            right = len(part) - 1

        for p in range(0, len(part)):
            i = 0
            direc = 0
            shadow = []
            while part[p].y + i * direction >= 0 and part[p].y + i * direction < len(output1) and output1[part[p].y + i * direction][part[p].x][0] < 250:
                if p == left:
                    direc = -1
                    shadow.append(pos(part[p].x, part[p].y + i * direction))
                elif p == right:
                    direc = 1
                    shadow.append(pos(part[p].x, part[p].y + i * direction))
                i += 1
            fit.append(pos(part[p].x, part[p].y + (i - 1) * direction))
            if direc != 0:
                for s in shadow:
                    i = 0
                    while s.x + i * direc < len(output1[0]) and s.x + i * direc >= 0 and output1[s.y][s.x + i * direc][0] < 250:
                        i += 1
                    fit.append(pos(s.x + (i - 1) * direc, s.y))
                shadow = []
    else:
        tmpx1 = midx
        tmpx2 = midx
        while tmpx1 >= 0 and output1[midy][tmpx1][0] < 250:
            tmpx1 -= 1
        while tmpx2 < len(output1[0]) and output1[midy][tmpx2][0] < 250:
            tmpx2 += 1
        direction = 1 if tmpx1 > tmpx2 else -1

        if a.y > b.y:
            up = len(part) - 1
            down = 0
        else:
            up = 0
            down = len(part) - 1

        for p in range(0, len(part)):
            i = 0
            direc = 0
            shadow = []
            while part[p].x + i * direction >= 0 and part[p].x + i * direction < len(output1[0]) and output1[part[p].y][part[p].x + i * direction][0] < 250:
                if p == up:
                    direc = -1
                    shadow.append(pos(part[p].x + i * direction, part[p].y))
                elif p == down:
                    direc = 1
                    shadow.append(pos(part[p].x + i * direction, part[p].y))
                i += 1
            fit.append(pos(part[p].x + i * direction, part[p].y))
            if direc != 0:
                for s in shadow:
                    i = 0
                    while s.y + i * direc >= 0 and s.y + i * direc < len(output1) and output1[s.y + i * direc][s.x][0] < 250:
                        i += 1
                    fit.append(pos(s.x, s.y + i * direc))
                shadow = []


def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def tri_test(a, b, c):
    s = [a, b, c]
    dist = [distance(a, b), distance(b, c), distance(c, a)]
    mx = 0
    for d in range(0, len(dist)):
        if dist[mx] < dist[d]:
            mx = d
    return [s[mx], s[(mx + 2)%3], s[(mx + 1)%3]]
    
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
output = np.zeros(imgm_arr.shape).astype(np.uint8)
map_road = copy.deepcopy(road)
print_road(output)
output1 = copy.deepcopy(output)
road = []
output = np.zeros(imgm_arr.shape).astype(np.uint8)
change(imgd_arr, 0)
get_road(output, 0)
get_road(output, 1)
draw_road = copy.deepcopy(road)
seq = get_sequence(20)
fitted = fit_on_map(seq)
for f in fitted:
    if f.y >= 0 and f.y < len(output1) and f.x >= 0 and f.x < len(output1[0]):
        output1[f.y][f.x] = [255, 255, 0]

print_road(output)


output2 = copy.deepcopy(output)

output = np.zeros(imgm_arr.shape).astype(np.uint8)
for p in map_road:
    output2[p.y][p.x][0] = 255
# find_path()

#print(output)
img = Image.fromarray(output1)
img.save("t1.png")
img = Image.fromarray(output2)
img.save("t2.png")
img = Image.fromarray(output)
img.save("t3.png")

"""
for i in range(0, len(seq)):
    for s in range(0, i):
        output[seq[i].y][seq[i].x] = [255, 0, 0]
    img = Image.fromarray(output)
    img.save("t" + str(i) + ".png")

for s in range(0, len(seq) - 1):
    print(distance(seq[s], seq[s + 1]))
"""

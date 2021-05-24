import numpy as np
from PIL import Image
import sys
import glob, os
import copy
import math
import argparse

np.set_printoptions(threshold=sys.maxsize)


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

def get_road(road, m, f):
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

def print_road(road, m):
    for r in road:
        for i in range(0, 2):
            if r.x + i < len(m[0]):
                m[r.y][r.x + i] = [255, 0, 0]
                    
def get_sequence(output, draw_road, detect_block):
    #seq = [blocks.pop(0)]
    seq = [draw_road.pop(0)]
    a = 0
    it = 0
    max_iter = 500
    while len(draw_road) > 0 and it < max_iter:
        it += 1
        print(len(draw_road))
        i = 0
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

def fit_on_map(output1, seq):
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
        blk = get_range(output1, s, detect_block)
        flag = False
        for i in range(blk[0], blk[1]):
            for j in range(blk[2], blk[3]):
                if output1[i][j][0] == 255:
                    flag = True
        if not flag:
            part.append(s)
        else:
            fit_road(output1, fit[len(fit) - 1], s, part, fit)
            fit.append(s)
            part = []

    for f in fit:
        if f.y >= 0 and f.y < len(output1) and f.x >= 0 and f.x < len(output1[0]):
            output1[f.y][f.x] = [255, 255, 0]

    return fit

def get_range(output, p, detect_block):
    up = p.y - detect_block if p.y - detect_block >= 0 else 0
    down = p.y + detect_block if p.y + detect_block < len(output) else len(output) - 1
    left = p.x - detect_block if p.x - detect_block >= 0 else 0
    right = p.x + detect_block if p.x + detect_block < len(output[0]) else len(output[0]) - 1
    return [up, down, left, right]

def fit_road(output1, a, b, part, fit):
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

def change(output, img_arr, f):
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


def fit_origin(output, map_arr, out_arr, detect_block):
    for i in range(0, len(output)):
        for j in range(0, len(output[0])):
            if output[i][j][0] == 255 and output[i][j][1] == 255:
                r = get_range(output, pos(j, i), detect_block)
                for k in range(r[0], r[1]):
                    for l in range(r[2], r[3]):
                        flag = True
                        for m in range(0, 3):
                            if map_arr[k][l][m] < 250:
                                flag = False
                        if flag:
                            map_arr[k][l] = [255, 0, 0]
                            out_arr[k][l] = [255, 0, 0]

def fix_size(arr):
    tmp = np.array([])
    if len(arr[0][0]) == 4:
        for i in range(0, len(arr)):
            for j in range(0, len(arr[0])):
                tmp = np.append(tmp, np.delete(arr[i][j], 3, 0))
        arr = np.resize(tmp, (len(arr), len(arr[0])))

def map_draw_fit(map_arr, draw_arr, out_arr):
    detect_block = 20
    output = np.zeros(map_arr.shape).astype(np.uint8)
    change(output, map_arr, 1)
    road = []
    get_road(road, output, 0)
    get_road(road, output, 1)
    output = np.zeros(map_arr.shape).astype(np.uint8)
    print_road(road, output)
    output_map = copy.deepcopy(output)
    road = []
    output = np.zeros(draw_arr.shape).astype(np.uint8)
    change(output, draw_arr, 0)
    get_road(road, output, 0)
    get_road(road, output, 1)
    seq = get_sequence(output, road, detect_block)
    fitted = fit_on_map(output_map, seq)
    fit_origin(output_map, map_arr, out_arr, 10)    

    return output_map

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default='ex1.png', help='input map file name')
    parser.add_argument('--draw', default='a1.png', help='input draw file name')
    parser.add_argument('--output', default='output.png', help='output file name')
    parser.add_argument('--binary_output', default='output.png', help='output file name')
    args = parser.parse_args()

    cur_path = os.path.dirname(__file__)
    map_path = os.path.relpath("./input/map/" + args.map, cur_path)
    draw_path = os.path.relpath("./input/draw/" + args.draw, cur_path)
    output_path = os.path.relpath("./output/" + args.output, cur_path)

    img_map = Image.open(map_path)
    map_size = img_map.size
    map_arr = np.array(img_map)
    fix_size(map_arr)
    draw_arr = np.array(Image.open(draw_path).resize(map_size))
    fix_size(draw_arr)
    binary_arr = np.zeros(map_arr.shape).astype(np.uint8)
    output = map_draw_fit(map_arr, draw_arr, binary_arr)
    img = Image.fromarray(map_arr)
    img.save(output_path)
    img = Image.fromarray(binary_arr)
    img.save("convert_binary/training_data/output/"+args.binary_output)

    """
    for i in range(0, len(seq)):
        for s in range(0, i):
            output[seq[i].y][seq[i].x] = [255, 0, 0]
        img = Image.fromarray(output)
        img.save("t" + str(i) + ".png")

    for s in range(0, len(seq) - 1):
        print(distance(seq[s], seq[s + 1]))
    """

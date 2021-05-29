import numpy as np
from PIL import Image
import sys
import glob, os
import copy
import argparse

np.set_printoptions(threshold=sys.maxsize)

class treenode:
    def __init__(self, child, pos):
        self.child = child
        self.pos = pos
        self.on_road = []
        self.parent = None

def get_sequence(draw_arr, map_arr):
    root = None
    for i in range(0, len(map_arr)):
        for j in range(0, len(map_arr[0])):
            if draw_arr[i][j][0] == 255:
                root = build_tree([i, j], root, map_arr, draw_arr)
                return root


def get_range(size, detect_block, pos):
    area = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
    area['up'] = pos[0] - detect_block if pos[0] - detect_block > 0 else 0
    area['down'] = pos[0] + detect_block if pos[0] + detect_block < size[0] else size[0] - 1
    area['left'] = pos[1] - detect_block if pos[1] - detect_block > 0 else 0
    area['right'] = pos[1] + detect_block if pos[1] + detect_block < size[1] else size[1] - 1
    return area

"""
     |
2    |    1
-----|-----
3    |    4
     |
"""

def detect_on_edge(i, j, area, mid, direction):
    if i == area['up']:
        direct = '2' if j < mid[1] else '1'
        direction[direct].append([i, j])
    elif i == area['down'] - 1:
        direct = '3' if j < mid[1] else '4'
        direction[direct].append([i, j])
    elif j == area['left']:
        direct = '2' if i < mid[0] else '3'
        direction[direct].append([i, j])
    elif j == area['right'] - 1:
        direct = '1' if i < mid[0] else '4'
        direction[direct].append([i, j])


def build_tree(start, root, map_arr, draw_arr):
    detect_block = 5
    root = treenode([], start)
    leaf = [root]
    while len(leaf) > 0:
        next_leaf = []
        for lf in leaf:
            direction = {'1': [], '2': [], '3': [], '4': []}
            area = get_range(map_arr.shape, detect_block, lf.pos)
            for i in range(area['up'], area['down']):
                for j in range(area['left'], area['right']):
                    if draw_arr[i][j][0] == 255:
                        if draw_arr[i][j][2] == 0:
                            detect_on_edge(i, j, area, lf.pos, direction)
                        draw_arr[i][j][2] = 255
            for key in direction:
                if len(direction[key]) > 0:
                    newnode = treenode([], direction[key][0])
                    for pix in direction[key]:
                        if map_arr[pix[0]][pix[1]][0] == 255:
                            newnode.on_road.append(pix)
                            # print
                            b = get_range(map_arr.shape, 5, pix)
                            for k in range(b['up'], b['down']):
                                for l in range(b['left'], b['right']):
                                    if map_arr[k][l][0] == 255:
                                        map_arr[k][l][1] = 255
                            #

                    lf.child.append(newnode)
                    next_leaf.append(newnode)
        leaf = next_leaf

    return root


def nearest_point(start, map_arr):

    # find nearest point for start
    candidate = {'up': start[0], 'down': start[0], 'left': start[1], 'right': start[1]}
    distance = {'up': 0, 'down': 0, 'left': 0, 'right': 0}

    while candidate['up'] > 0 and map_arr[candidate['up']][start[1]][0] == 0:
        candidate['up'] -= 1
        distance['up'] += 1
    while candidate['down'] < len(map_arr) - 1 and map_arr[candidate['down']][start[1]][0] == 0:
        candidate['down'] += 1
        distance['down'] += 1
    while candidate['left'] > 0 and map_arr[start[0]][candidate['left']][0] == 0:
        candidate['left'] -= 1
        distance['left'] += 1
    while candidate['right'] < len(map_arr[0]) - 1 and map_arr[start[0]][candidate['right']][0] == 0:
        candidate['right'] += 1
        distance['right'] += 1

    min_dist = 'up'
    for key in distance:
        if distance[key] < distance[min_dist]:
            min_dist = key

    if min_dist == 'up' or min_dist == 'down':
        return [candidate[min_dist], start[1]]
    else:
        return [start[0], candidate[min_dist]]


def fit_on_map(draw_arr, map_arr):

    root = get_sequence(draw_arr, map_arr)
    start = nearest_point(root.pos, map_arr)
    traverse_tree(start, root, [], map_arr)
    
def traverse_tree(root_pos, parent, start, map_arr):
    if len(start) == 0:
        start = root_pos
    for ch in parent.child:
        if len(ch.on_road) > 0:
            shortest_path(start, ch.on_road[0], map_arr)
            traverse_tree(root_pos, ch, ch.on_road[0], map_arr)
        else:
            traverse_tree(root_pos, ch, start, map_arr)


def shortest_path(start, end, map_arr):
    print("hi")
    detect_block = 5
    detect_end = 3
    root = treenode([], start)
    leaf = [root]
    max_iter = 1000
    count = 0
    while len(leaf) > 0 and count < max_iter:
        count += 1
        print(count)
        next_leaf = []
        for lf in leaf:
            direction = {'1': [], '2': [], '3': [], '4': []}
            area = get_range(map_arr.shape, detect_block, lf.pos)
            for i in range(area['up'], area['down']):
                for j in range(area['left'], area['right']):
                    if map_arr[i][j][0] == 255:
                        if map_arr[i][j][2] == 0:
                            detect_on_edge(i, j, area, lf.pos, direction)
                        map_arr[i][j][2] = 1
            for key in direction:
                if len(direction[key]) > 0:
                    for pix in direction[key]:
                        area = get_range(map_arr.shape, detect_end, pix)
                        for i in range(area['up'], area['down']):
                            for j in range(area['left'], area['right']):
                                if i == end[0] and j == end[1]:
                                    # find path and print
                                    node = lf
                                    while node:
                                        bk = get_range(map_arr.shape, detect_block, node.pos)
                                        for i in range(bk['up'], bk['down']):
                                            for j in range(bk['left'], bk['right']):
                                                if map_arr[i][j][0] == 255:
                                                    map_arr[i][j][1] = 255
                                        node = node.parent
                                    # reset
                                    for i in range(0, len(map_arr)):
                                        for j in range(0, len(map_arr[0])):
                                            map_arr[i][j][2] = 0;
                                    return

                    newnode = treenode([], direction[key][0])
                    newnode.parent = lf
                    lf.child.append(newnode)
                    next_leaf.append(newnode)
        leaf = next_leaf
    


def convert_binary(image, ismap):    
    output = np.zeros(image.shape).astype(np.uint8)
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if ismap and image[i][j][0] > 250 and image[i][j][1] > 250 and image[i][j][2] > 250:
                output[i][j][0] = 255
            elif not ismap and (image[i][j][0] < 250 or image[i][j][1] < 250 or image[i][j][2] < 250):
                output[i][j][0] = 255
    return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default='m1.png', help='input map file name')
    parser.add_argument('--draw', default='d1.png', help='input draw file name')
    parser.add_argument('--output', default='output.png', help='output file name')
    args = parser.parse_args()

    cur_path = os.path.dirname(__file__)
    map_path = os.path.relpath("./input/map/" + args.map, cur_path)
    draw_path = os.path.relpath("./input/draw/" + args.draw, cur_path)
    output_path = os.path.relpath("./output/" + args.output, cur_path)

    size = [1014, 570, 3]

    map_array = np.array(Image.open(map_path).resize(size[0:2]))
    draw_array = np.array(Image.open(draw_path).resize(size[0:2]))
    map_binary = convert_binary(map_array, True)
    draw_binary = convert_binary(draw_array, False)

    fit_on_map(draw_binary, map_binary)
    # Image.fromarray(draw_binary).save("tmp.png")
    Image.fromarray(map_binary).save(args.output)

    

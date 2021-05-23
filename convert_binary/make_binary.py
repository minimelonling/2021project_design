import numpy as np
from PIL import Image
import sys
import glob, os
import argparse
import shutil

np.set_printoptions(threshold=sys.maxsize)


def mapinner(array):
    for a in array:
        if a < 250: # if one of RGB less than 250, it should not be white.
            return False # the pixel is part of the road.
    return True

def drawinner(array):
    for a in array:
        if a < 250:
            return True # the pixel is part of the draw.
    return False


def change(image_array, map_draw_flag, binary_flag, output):
    """
    if map_draw_flag = 1, pass map array; otherwise, pass draw array.
    if binary_flag = 1, show binary result; otherwise, show red green result.
    """
    for i in range(0, image_array.shape[0]):
        for j in range(0, image_array.shape[1]):
            if map_draw_flag:
                if mapinner(image_array[i][j]):
                    output[i][j][0] = 1 if binary_flag == 1 else 255
            else:
                if drawinner(image_array[i][j]):
                    output[i][j][1] = 1 if binary_flag == 1 else 255

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', default = '1')
    parser.add_argument('--pairwise', default = '1')
    parser.add_argument('--base', default = '1')
    args = parser.parse_args()

    i = 0
    size = (500, 500, 3) # output image size

    # pairwise generate binary picture
    if args.pairwise == "1":
        for mapfile, drawfile in zip(os.listdir("map"), os.listdir("draw")):
            if mapfile.endswith(".png") and drawfile.endswith(".png"):
                map_path = os.path.join("map/", mapfile)
                draw_path = os.path.join("draw/", drawfile)
                map_array = np.array(Image.open(map_path).resize(size[0:2]))
                draw_array = np.array(Image.open(draw_path).resize(size[0:2]))
                output = np.zeros(size).astype(np.uint8)
                change(map_array, 1, int(args.binary), output)
                change(draw_array, 0, int(args.binary), output)
                image = Image.fromarray(output)
                image.save(str(i+int(args.base))+".png")
                shutil.move(str(i+int(args.base))+".png", "training_data/input");
                i += 1
    else:
        for mapfile in os.listdir("map"):
            if mapfile.endswith(".png"):
                map_path = os.path.join("map/", mapfile)
                map_array = np.array(Image.open(map_path).resize(size[0:2]))
                for drawfile in os.listdir("draw"):
                    if drawfile.endswith(".png"):
                        draw_path = os.path.join("draw/", drawfile)
                        draw_array = np.array(Image.open(draw_path).resize(size[0:2]))
                        output = np.zeros(size).astype(np.uint8)
                        change(map_array, 1, int(args.binary), output)
                        change(draw_array, 0, int(args.binary), output)
                        image = Image.fromarray(output)
                        image.save(str(i+int(args.base))+".png")
                        shutil.move(str(i+int(args.base))+".png", "training_data/input");
                        i += 1
                

import os
import sys
import geopy.distance
from PIL import Image

def combineAB(A, B, counter):
    images = list(map(Image.open, [A, B]))
    widths, heights = zip(*(i.size for i in images))
    print(widths[0], heights[0])

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        print("im", im)
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    desRoot = '../datasets//home/rui/Dev/Projects/semantic-segmentation-pytorch/data/results_1/'
    if not os.path.exists(desRoot):
        os.makedirs(desRoot)

    a1 = A.split("/")[-1].split(".jpg")[0].split("_")[1]
    a2 = A.split("/")[-1].split(".jpg")[0].split("_")[2]
    b1 = B.split("/")[-1].split(".jpg")[0].split("_")[1]
    b2 = B.split("/")[-1].split(".jpg")[0].split("_")[2]

    a_co = (a1, a2)
    b_co = (b1, b2)
    dis = geopy.distance.vincenty(a_co, b_co).m

    new_im.save('{}{}_{}_.jpg'.format(desRoot, counter, dis))


rootdir = './datasets/streetview_seg_car_in4/'

prevSubDir, A, B, prevFile = "", "", "", ""
current, prev = "", ""

counter = 1

testing = False

print(os.getcwd())

for filename in os.listdir(rootdir):
    print(filename)

    image = Image.open(os.getcwd() + '/datasets/streetview_seg_car_in4/' + filename).convert('RGB')

    w, h = image.size
    w2 = int(w / 2)
    w4 = int(w / 4)
    A = image.crop((w4, 0, w2, h))
    print(filename)

    desRoot = './datasets/streetview_car_real/'
    if not os.path.exists(desRoot):
        os.makedirs(desRoot)

    A.save('{}{}'.format(desRoot, filename))



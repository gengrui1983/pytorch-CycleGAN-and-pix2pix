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

    desRoot = '../datasets/new_streetview/'
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


rootdir = '../datasets/streetview_jpg/'

prevSubDir, A, B, prevFile = "", "", "", ""
current, prev = "", ""

counter = 1

testing = False

for subdir, dirs, files in os.walk(rootdir):
    dirs.sort(reverse=True)

    if testing and counter > 10: break

    files.sort(key=lambda x: int(x.split("_")[0]))
    for file in files:
        sameDir = prevFile != "" and prevSubDir == subdir

        if sameDir:
            current = os.path.join(subdir, file)
            prev = os.path.join(subdir, prevFile)

            combineAB(prev, current, counter)
            counter += 1
            if testing and counter == 10: break
        prevSubDir = subdir
        prevFile = file

        print("A: {}, B: {}".format(current, prev))

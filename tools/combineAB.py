import os
import sys
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

    new_im.save('{}{}.jpg'.format(desRoot, counter))


rootdir = '../datasets/streetview_jpg/'

prevSubDir, A, B, prevFile = "", "", "", ""
current, prev = "", ""

counter = 1

testing = False

for subdir, dirs, files in os.walk(rootdir):
    dirs.sort(reverse=True)

    if testing and counter > 10: break
    for file in sorted(files, reverse=True):
        sameDir = prevFile != "" and prevSubDir == subdir

        if sameDir:
            current = os.path.join(subdir, file)
            prev = os.path.join(subdir, prevFile)

            combineAB(current, prev, counter)
            counter += 1
            if testing and counter == 10: break
        prevSubDir = subdir
        prevFile = file

        print("A: {}, B: {}".format(current, prev))

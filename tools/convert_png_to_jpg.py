import os
import sys
from PIL import Image


def convert(file, subdir, filename):
    im = Image.open(file)
    rgb_im = im.convert('RGB')

    newFilename = filename.replace("png", "jpg")

    dir = '../datasets/streetview_jpg/{}'.format(subdir)

    if not os.path.exists(dir):
        os.makedirs(dir)

    rgb_im.save('../datasets/streetview_jpg/{}/{}'.format(subdir, newFilename))


rootdir = '../datasets/streetview/'

prevSubDir, A, B, prevFile = "", "", "", ""
current, prev = "", ""

counter = 1

for subdir, dirs, files in os.walk(rootdir):

    folderName = subdir.split("/")[3]

    print(folderName)
    # if counter > 10: break
    for file in files:
        f = os.path.join(subdir, file)
        convert(f, folderName, file)

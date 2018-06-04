import os
import sys
import geopy.distance
from PIL import Image

originals_dir = './datasets/streetview_seg_car/images/'
segs_dir = './datasets/streetview_seg_car/segs/'
real_dir = './datasets/streetview_seg_car/real/'

def combineABC(A, B, C, filename):
    images = list(map(Image.open, [A, B, C]))
    widths, heights = zip(*(i.size for i in images))
    print(widths[0], heights[0])

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        print("x_offset", x_offset)
        print("im", im)
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    desRoot = './datasets/new_streetview_with_seg/'
    if not os.path.exists(desRoot):
        os.makedirs(desRoot)

    new_im.save('{}{}.jpg'.format(desRoot, filename))


print(os.getcwd())

for filename in os.listdir(real_dir):
    print(filename)

    image_file = filename.replace(".jpg", "_fake_B.png")
    image = os.getcwd() + '/datasets/streetview_seg_car/images/' + image_file

    seg = os.getcwd() + '/datasets/streetview_seg_car/segs/' + image_file

    image_file_real_B = filename.replace(".jpg", "_real_B.png")
    real = os.getcwd() + '/datasets/streetview_seg_car/images/' + image_file_real_B

    combineABC(image, seg, real, filename)



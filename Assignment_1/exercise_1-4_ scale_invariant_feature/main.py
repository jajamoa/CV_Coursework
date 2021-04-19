from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from generate_gaussian_pyramid import generate_gaussian_pyramid
from generate_DoG_pyramid import generate_DoG_pyramid
from get_keypoints import get_keypoints
import warnings

warnings.filterwarnings('ignore')


def main(args):
    """[main] main function of URLAggregate tool

    Args:
        args: user input parameters

    """

    im = Image.open(args.input)
    im_1 = im.resize((im.size[0]//3, im.size[1]//3))
    im_2 = im.resize((im.size[0]//2, im.size[1]//2))
    img_1 = np.array(im_1.convert('L'))
    img_2 = np.array(im_2.convert('L'))
    g_pyr_1 = generate_gaussian_pyramid(img_1)
    g_pyr_2 = generate_gaussian_pyramid(img_2)
    d_pyr_1 = generate_DoG_pyramid(g_pyr_1)
    d_pyr_2 = generate_DoG_pyramid(g_pyr_2)
    kp_pyr_1 = get_keypoints(d_pyr_1)
    kp_pyr_2 = get_keypoints(d_pyr_2)

    im_1 = im.resize((im.size[0]//3, im.size[1]//3))
    draw = ImageDraw.Draw(im_1)
    scale = 1
    for pyr in kp_pyr_1:
        for x, y, s in pyr:
            x, y, s = x*scale, y*scale, s*scale
            if s <= 3:
                continue
            x0, x1 = x-s, x+s
            y0, y1 = y-s, y+s
            draw.arc((x0, y0, x1, y1), start=0, end=360, fill='red', width=1)
        scale *= 2
    plt.imshow(im_1, cmap='gray', vmin=0, vmax=255)
    plt.savefig(os.path.join(args.output, 'output_1.jpg'), dpi=300)
    print(
        f"[Saving...] Saved the image to {os.path.join(args.output, 'output_1.jpg')}")
    plt.show()
    print('[Done]')

    im_2 = im.resize((im.size[0]//2, im.size[1]//2))
    draw = ImageDraw.Draw(im_2)
    scale = 1
    for pyr in kp_pyr_2:
        for x, y, s in pyr:
            x, y, s = x*scale, y*scale, s*scale
            if s <= 3:
                continue
            x0, x1 = x-s, x+s
            y0, y1 = y-s, y+s
            draw.arc((x0, y0, x1, y1), start=0, end=360, fill='red', width=1)
        scale *= 2
    plt.imshow(im_2, cmap='gray', vmin=0, vmax=255)
    plt.savefig(os.path.join(args.output, 'output_2.jpg'), dpi=300)
    print(
        f"[Saving...] Saved the image to {os.path.join(args.output, 'output_2.jpg')}")
    plt.show()
    print('[Done]')


if __name__ == '__main__':
    if ('-?' in sys.argv):
        sys.argv.append('--help')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-i', '--input', type=str, default=r'input.jpg',
                        help='input image path, default is ''input.jpg''')
    parser.add_argument('-o', '--output', type=str, default=r'.',
                        help='folder path to export the output images, default is current folder')
    args = parser.parse_args()
    main(args)

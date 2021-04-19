import numpy as np
from scipy.ndimage.filters import convolve


def gaussian_filter(sigma):
    """[gaussian_filter] function to generate sigma filter of specific sigma

    Args:
        sigma: sigma

    Return:
        gaussian filter kernel
    """
    size = 2*np.ceil(3*sigma)+1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()


def generate_octave(init_level, s, sigma):
    """[generate_octave] function to generate octaves

    Args:
        init_level: init image
        s: num of img per octave
        sigma: sigma of gaussian kernel

    Return:
        octave: gaussian octave
    """
    octave = [init_level]
    k = 2**(1/s)
    kernel = gaussian_filter(k * sigma)
    for _ in range(s+2):
        next_level = convolve(octave[-1], kernel)
        octave.append(next_level)
    return octave


def generate_gaussian_pyramid(im, num_octave=3, s=3, sigma=1.6):
    """[generate_gaussian_pyramid] function to generate gaussian pyramid

    Args:
        im: init image
        num_octave: num of octaves
        s: num of img per octave
        sigma: sigma of gaussian kernel

    Return:
        pyr: gaussian pyramid
    """
    pyr = []
    for _ in range(num_octave):
        octave = generate_octave(im, s, sigma)
        pyr.append(octave)
        im = octave[-3][::2, ::2]
    return pyr

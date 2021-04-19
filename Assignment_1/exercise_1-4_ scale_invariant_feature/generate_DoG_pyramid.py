import numpy as np


def generate_DoG_octave(gaussian_octave):
    """[generate_DoG_octave] function to generate DoG octave from gaussian octave

    Args:
        gaussian_octave: gaussian octave

    Return:
        DoG_octave: DoG octave
    """
    octave = []
    for i in range(1, len(gaussian_octave)):
        octave.append(gaussian_octave[i] - gaussian_octave[i-1])
    return np.concatenate([o[:, :, np.newaxis] for o in octave], axis=2)


def generate_DoG_pyramid(gaussian_pyramid):
    """[generate_DoG_pyramid] function to generate DoG pyramid from gaussian pyramid

    Args:
        gaussian_pyramid: gaussian pyramid

    Return:
        pyr: DoG pyramid
    """
    pyr = []
    for gaussian_octave in gaussian_pyramid:
        pyr.append(generate_DoG_octave(gaussian_octave))
    return pyr

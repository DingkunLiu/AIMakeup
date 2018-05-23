import numpy as np
import scipy.interpolate as si


def mask_interpolation(landmarks):
    n = landmarks.shape[0]
    x = landmarks[:, 0]
    y = landmarks[:, 1]

    t = np.arange(0, n)

    num_interp = 3 * n
    ipl_t = np.linspace(0, n-1, num_interp)

    x_tup = si.splrep(t, x, k=3)
    y_tup = si.splrep(t, y, k=3)

    x_list = list(x_tup)
    xl = x.tolist()
    x_list[1] = xl + [0] * 4

    y_list = list(y_tup)
    yl = y.tolist()
    y_list[1] = yl + [0] * 4

    x_i = si.splev(ipl_t, x_list)
    y_i = si.splev(ipl_t, y_list)

    points = np.stack((x_i, y_i), axis=1)


    return points


def adjustContrast(src, contrast):
    width = src.shape[0]
    height = src.shape[1]
    mean = src.mean(axis=(0, 1))
    if contrast <= -255:
        dst = np.tile(mean, [width, height, 1])
    elif -255 < contrast <= 0:
        dst = src + (src - mean) * contrast / 255
    elif 0 < contrast < 255:
        contrast = 255 * 255 / (255 - contrast) - 255
        dst = src + (src - mean) * contrast / 255
    else:
        dst = np.where(src > mean, 255, 0)
    return dst


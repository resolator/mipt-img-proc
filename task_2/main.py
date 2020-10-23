#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Script for fractal encode/decode."""
import cv2
import argparse

import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed


ANGLE_CODE = {1: cv2.ROTATE_90_CLOCKWISE,
              2: cv2.ROTATE_180,
              3: cv2.ROTATE_90_COUNTERCLOCKWISE}

# flip direction, angle code
TRANS_CODE = [
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (-1, 0),
    (-1, 1),
    (-1, 2),
    (-1, 3),
]


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img-path', required=True,
                        help='Path to input image.')
    parser.add_argument('--save-to', required=True,
                        help='Path to save image.')
    parser.add_argument('--block-size', type=int, default=4, choices=[4, 8],
                        help='Block size for patterns.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of threads to encode.')
    parser.add_argument('--iters', type=int, default=10,
                        help='Restoring iterations.')
    parser.add_argument('--shrink', action='store_true',
                        help='Shrink original image before work (speed up).')

    return parser.parse_args()


def rotate(img, angle):
    """OpenCV rotation wrapper."""
    return cv2.rotate(img, ANGLE_CODE[angle]) if angle != 0 else img


def apply_transform(img, trans_code, brightness=0.0, contrast=1.0):
    """Apply given transform to block.

    Parameters
    ----------
    img : numpy.ndarray
        Image or (block of an image).
    trans_code : int
        Index for one of TRANS_CODE value defining transform.
    brightness : float
        Calculated brightness value.
    contrast : float
        Calculated contrast value.

    Returns
    -------
    numpy.ndarray
        Transformed image.

    """
    flip_dir, angle = TRANS_CODE[trans_code]
    return brightness + contrast * rotate(img[:, ::flip_dir], angle)


def shrink(img, win_size):
    """Decrease image size by calculating window mean."""
    res = np.zeros((img.shape[0] // win_size, img.shape[1] // win_size))

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = np.mean(img[i * win_size:(i + 1) * win_size,
                                    j * win_size:(j + 1) * win_size])

    return res


def calc_brightness_contrast(src, dst):
    src = np.concatenate((np.ones((src.size, 1)),
                          src.reshape((-1, 1))), axis=1)
    dst = dst.reshape(-1)

    return np.linalg.lstsq(src, dst, rcond=None)[0]


def create_blocks(img, block_sz):
    src_sz = block_sz * 2

    blocks = []
    for i in range((img.shape[0] - src_sz) // src_sz + 1):
        for j in range((img.shape[1] - src_sz) // src_sz + 1):
            block = img[i * src_sz:(i + 1) * src_sz,
                        j * src_sz:(j + 1) * src_sz]
            shrinked = shrink(block, 2)

            for trans_code in range(len(TRANS_CODE)):
                blocks.append(
                    [i, j, trans_code, apply_transform(shrinked, trans_code)])

    return blocks


def calc_l2(img_1, img_2):
    return np.square(np.subtract(img_1, img_2)).mean()


def make_job(shape_1, i, img, dst_sz, blocks):
    transforms = np.zeros((1, shape_1, 5), dtype=float)
    for j in range(shape_1):
        min_err = np.inf
        dst_block = img[i * dst_sz:(i + 1) * dst_sz,
                        j * dst_sz:(j + 1) * dst_sz]

        # Select best transform
        for y, x, trans_code, block in blocks:
            brightness, contrast = calc_brightness_contrast(
                block, dst_block)
            block = contrast * block + brightness

            err = calc_l2(block, dst_block)
            if err < min_err:
                transforms[0, j] = (y, x, trans_code, contrast, brightness)
                min_err = err

    return transforms


def fractal_encode(img, dst_sz, workers=1):
    blocks = create_blocks(img, dst_sz)

    p = Parallel(n_jobs=workers)
    res = p(delayed(make_job)(img.shape[1] // dst_sz, i, img, dst_sz, blocks)
            for i in tqdm(range(img.shape[0] // dst_sz), desc='Encoding'))

    transforms = np.vstack(res)

    return transforms


def restore_step(img, transforms, block_sz):
    res = np.zeros((transforms.shape[0] * block_sz,
                    transforms.shape[1] * block_sz))

    for i in range(transforms.shape[0]):
        for j in range(transforms.shape[1]):
            y, x, trans_code, contrast, brightness = transforms[i, j]
            y, x, trans_code = int(y), int(x), int(trans_code)

            src_block = shrink(img[y * block_sz * 2:(y + 1) * block_sz * 2,
                                   x * block_sz * 2:(x + 1) * block_sz * 2], 2)
            dst = apply_transform(src_block, trans_code, brightness, contrast)
            res[i * block_sz:(i + 1) * block_sz,
                j * block_sz:(j + 1) * block_sz] = dst

    return res


def fractal_decode(transforms, dst_sz, iters=8):
    iterations = [np.random.randint(0, 256, (transforms.shape[0] * dst_sz,
                                             transforms.shape[1] * dst_sz))]

    for _ in tqdm(range(iters), desc='Restoring iteration'):
        iterations.append(restore_step(iterations[-1], transforms, dst_sz))

    return iterations[-1]


def main():
    """Application entry point."""
    args = get_args()

    # read
    img = cv2.imread(args.img_path, cv2.IMREAD_GRAYSCALE)

    # process
    if args.shrink:
        img = shrink(img, 4)

    transforms = fractal_encode(img, args.block_size, args.workers)
    res = fractal_decode(transforms, args.block_size, args.iters)

    # save
    cv2.imwrite(args.save_to, res)


if __name__ == '__main__':
    main()

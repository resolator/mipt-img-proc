#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""OTSU binarization implementation with improvements."""
import cv2
import time
import argparse

import numpy as np

from tqdm import tqdm
from pathlib import Path
from functools import partial


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img-path', type=Path, nargs='+', required=True,
                        help='Path to image to binarize.')
    parser.add_argument('--save-to', type=Path, required=True,
                        help='Path to save dir.')
    parser.add_argument('--bs', type=int,
                        help='Pass this arg to use blocks binarization. '
                             'Defines block size. Recommended: 45.')

    return parser.parse_args()


def calc_otsu_th(img):
    """Calculate threshold for binarization using OTSU method."""
    bins = np.unique(img)

    bins = np.append(bins, img.max())
    hist, ths = np.histogram(img, bins=bins)
    ths = ths[:-1]

    img_sum = np.sum(img)
    best_sigma, best_th, cum_sum_1, pixels_num = [0.0] * 4

    for idx, th in enumerate(ths):
        pixels_num += hist[idx]
        cum_sum_1 += th * hist[idx]
        if pixels_num == 0 or (img.size - pixels_num) == 0:
            continue

        mean_1 = cum_sum_1 / pixels_num
        mean_2 = (img_sum - cum_sum_1) / (img.size - pixels_num)
        mean_diff = mean_1 - mean_2

        prob_1 = pixels_num / img.size
        prob_2 = 1 - prob_1

        sigma = prob_1 * prob_2 * mean_diff**2
        if sigma > best_sigma:
            best_sigma = sigma
            best_th = th

    return int(best_th)


def bin_otsu(img):
    """Apply OTSU binarization."""
    th = calc_otsu_th(img)
    img[img > th] = 255
    img[img <= th] = 0

    return img


def bin_blocks(img, bs=45):
    """Binarize image using OTSU for each block."""
    # expand image to needable size
    top, bot = 0, 0
    height_diff = img.shape[0] % bs
    if height_diff != 0:
        need_for_fit = bs - height_diff
        top = need_for_fit // 2
        bot = np.ceil(need_for_fit / 2).astype(int)

    left, right = 0, 0
    width_diff = img.shape[1] % bs
    if width_diff != 0:
        need_for_fit = bs - width_diff
        left = need_for_fit // 2
        right = np.ceil(need_for_fit / 2).astype(int)

    img = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_REFLECT)

    # binarizing by blocks
    for i in range(0, img.shape[0] - bs + 1, bs):
        for j in range(0, img.shape[1] - bs + 1, bs):
            block = img[i:i + bs, j:j + bs].copy()

            # heuristic fill
            blurred = cv2.bilateralFilter(block, 7, 75, 75)
            if np.var(blurred) < 25:
                img[i:i + bs, j:j + bs] = 255 if np.mean(block) >= 127 else 0
            else:
                th = calc_otsu_th(block)
                img[i:i + bs, j:j + bs][block > th] = 255
                img[i:i + bs, j:j + bs][block <= th] = 0

    # revert expanding
    if bot == 0:
        bot = -img.shape[0]

    if right == 0:
        right = -img.shape[1]

    return img[top:-bot, left:-right]


def main():
    """Application entry point."""
    args = get_args()
    args.save_to.mkdir(parents=True, exist_ok=True)

    spent_time = 0.0
    binarize_f = bin_otsu if args.bs is None else partial(bin_blocks,
                                                          bs=args.bs)
    for img_path in tqdm(args.img_path, desc='Binarizing images'):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        start_time = time.time()
        img = binarize_f(img)
        spent_time += time.time() - start_time

        save_path = args.save_to.joinpath(img_path.stem + '.png')
        cv2.imwrite(str(save_path), img)

    print('Algorithm working time for given images is {} seconds'.format(
        round(spent_time, 4)))


if __name__ == '__main__':
    main()

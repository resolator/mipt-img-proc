#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Script for PSNR calculation between two images."""
import cv2
import argparse

import numpy as np


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src-img',
                        help='Path to source image.')
    parser.add_argument('--dst-img',
                        help='Path to generated image.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    src_img = cv2.imread(args.src_img)
    dst_img = cv2.imread(args.dst_img)

    assert src_img.shape == dst_img.shape, 'shape mismatch'

    # to gray
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

    divider = src_img.shape[0] * src_img.shape[1]
    mse = np.power(src_gray - dst_gray, 2).sum() / divider

    psnr = 10 * np.log10(255**2 / mse)
    print('PSNR:', psnr)


if __name__ == '__main__':
    main()

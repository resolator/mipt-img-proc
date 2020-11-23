#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Script for PSNR calculation between two images."""
import cv2
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src-img', type=Path, required=True,
                        help='Path to source image.')
    parser.add_argument('--dst-img', type=Path, nargs='+', required=True,
                        help='Path to generated image.')

    return parser.parse_args()


def main():
    """Application entry point."""
    args = get_args()

    src_img = cv2.imread(str(args.src_img), cv2.IMREAD_GRAYSCALE)

    # calculate
    psnrs = []
    for dst_path in args.dst_img:
        dst_img = cv2.imread(str(dst_path), cv2.IMREAD_GRAYSCALE)
        assert src_img.shape == dst_img.shape, 'shape mismatch'

        divider = src_img.shape[0] * src_img.shape[1]
        mse = np.power(src_img - dst_img, 2).sum() / divider

        psnr = 10 * np.log10(255**2 / mse)
        psnrs.append([int(dst_path.stem.split('_')[-1]), psnr])

    # draw
    psnrs.sort()
    plt.title('PSNR for ' + args.src_img.name)
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.plot([x[0] for x in psnrs], [x[1] for x in psnrs])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

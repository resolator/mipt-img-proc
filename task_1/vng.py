#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Demosaicing realization of VNG for "Image processing" course."""
import cv2
import argparse

import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src-img', required=True,
                        help='Path to source GRAY image.')
    parser.add_argument('--save-to', required=True,
                        help='Path to save img result.')

    return parser.parse_args()


def calc_green_center_pixel(px, red_row=False):
    """Calculate full pixel with green center.

    Parameters
    ----------
    px : numpy.array
        Array of pixels with size 25 (5x5 flatten). First pixel is green.
    red_row : bool
        Set True if the first row contains red pixels instead of blue.

    Returns
    -------
    numpy.array
        BGR array for center pixel.

    """
    # calc initial diffs
    difs = {
        'r2_r5': np.abs(px[7] - px[17]),
        'g2_g7': np.abs(px[2] - px[12]),
        'g4_g9': np.abs(px[6] - px[16]) / 2,
        'g5_g10': np.abs(px[8] - px[18]) / 2,
        'b1_b3': np.abs(px[1] - px[11]) / 2,
        'b2_b4': np.abs(px[3] - px[13]) / 2,
        'b3_b4': np.abs(px[11] - px[13]),
        'g7_g8': np.abs(px[12] - px[14]),
        'g4_g5': np.abs(px[6] - px[8]) / 2,
        'g9_g10': np.abs(px[16] - px[18]) / 2,
        'r2_r3': np.abs(px[7] - px[9]) / 2,
        'r5_r6': np.abs(px[17] - px[19]) / 2,
        'g7_g12': np.abs(px[12] - px[22]),
        'b3_b5': np.abs(px[11] - px[21]) / 2,
        'b4_b6': np.abs(px[13] - px[23]) / 2,
        'g6_g7': np.abs(px[10] - px[12]),
        'r1_r2': np.abs(px[5] - px[7]) / 2,
        'r4_r5': np.abs(px[15] - px[17]) / 2,
        'g5_g9': np.abs(px[8] - px[16]),
        'g3_g7': np.abs(px[4] - px[12]),
        'b2_b3': np.abs(px[3] - px[11]),
        'r3_r5': np.abs(px[9] - px[17]),
        'g4_g10': np.abs(px[6] - px[18]),
        'g7_g13': np.abs(px[12] - px[24]),
        'b3_b6': np.abs(px[11] - px[23]),
        'r2_r6': np.abs(px[7] - px[19]),
        'g1_g7': np.abs(px[0] - px[12]),
        'b1_b4': np.abs(px[1] - px[13]),
        'r1_r5': np.abs(px[5] - px[17]),
        'g7_g11': np.abs(px[12] - px[20]),
        'b4_b5': np.abs(px[13] - px[21]),
        'r2_r4': np.abs(px[7] - px[15]),
    }

    # calc grads
    g_n = (difs['r2_r5'] + difs['g2_g7'] + difs['g4_g9'] +
           difs['g5_g10'] + difs['b1_b3'] + difs['b2_b4'])
    g_e = (difs['b3_b4'] + difs['g7_g8'] + difs['g4_g5'] +
           difs['g9_g10'] + difs['r2_r3'] + difs['r5_r6'])
    g_s = (difs['r2_r5'] + difs['g7_g12'] + difs['g4_g9'] +
           difs['g5_g10'] + difs['b3_b5'] + difs['b4_b6'])
    g_w = (difs['b3_b4'] + difs['g6_g7'] + difs['g4_g5'] +
           difs['g9_g10'] + difs['r1_r2'] + difs['r4_r5'])
    g_ne = difs['g5_g9'] + difs['g3_g7'] + difs['b2_b3'] + difs['r3_r5']
    g_se = difs['g4_g10'] + difs['g7_g13'] + difs['b3_b6'] + difs['r2_r6']
    g_nw = difs['g4_g10'] + difs['g1_g7'] + difs['b1_b4'] + difs['r1_r5']
    g_sw = difs['g5_g9'] + difs['g7_g11'] + difs['b4_b5'] + difs['r2_r4']

    grads = np.array([g_n, g_e, g_s, g_w, g_ne, g_se, g_nw, g_sw])
    directions = np.array(['n', 'e', 's', 'w', 'ne', 'se', 'nw', 'sw'])

    # get thresholded grads
    min_grad = grads.min()
    max_grad = grads.max()
    k1 = 1.5
    k2 = 0.5
    th = k1 * min_grad + k2 * (max_grad + min_grad)

    grads_mask = grads < th
    filterred_directions = directions[grads_mask]

    if len(filterred_directions) == 0:
        filterred_directions = directions

    # process directions
    r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
    for direction in filterred_directions:
        if direction[0] == 'n':
            if len(direction) == 1:
                r_sum += px[7]
                g_sum += (px[2] + px[12]) / 2
                b_sum += (px[1] + px[3] + px[11] + px[13]) / 4
            elif direction == 'ne':
                r_sum += (px[7] + px[9]) / 2
                g_sum += px[8]
                b_sum += (px[3] + px[13]) / 2
            else:
                r_sum += (px[5] + px[7]) / 2
                g_sum += px[6]
                b_sum += (px[1] + px[11]) / 2

        elif direction[0] == 's':
            if len(direction) == 1:
                r_sum += px[17]
                g_sum += (px[12] + px[22]) / 2
                b_sum += (px[11] + px[13] + px[21] + px[23]) / 4
            elif direction == 'se':
                r_sum += (px[17] + px[19]) / 2
                g_sum += px[18]
                b_sum += (px[13] + px[23]) / 2
            else:
                r_sum += (px[15] + px[17]) / 2
                g_sum += px[16]
                b_sum += (px[11] + px[21]) / 2

        elif direction[0] == 'e':
            r_sum += (px[7] + px[9] + px[17] + px[19]) / 4
            g_sum += (px[12] + px[14]) / 2
            b_sum += px[13]
        else:
            r_sum += (px[5] + px[7] + px[15] + px[17]) / 4
            g_sum += (px[10] + px[12]) / 2
            b_sum += px[11]

    r = np.clip(px[12] + (r_sum - g_sum) / len(filterred_directions), 0, 255)
    b = np.clip(px[12] + (b_sum - g_sum) / len(filterred_directions), 0, 255)

    if red_row:
        b, r = r, b

    return np.array([b, px[12], r], dtype=np.uint8)


def calc_red_blue_center_pixel(px, first_blue=False):
    """Calculate full pixel with blue or red center.

    Parameters
    ----------
    px : numpy.array
        Array of pixels with size 25 (5x5 flatten). First pixel is red or blue.
    first_blue : bool
        Set True if the first pixel is blue instead of red.

    Returns
    -------
    numpy.array
        BGR array for center pixel.

    """
    # calc initial diffs
    difs = {
        'g4_g9': np.abs(px[7] - px[17]),
        'r2_r5': np.abs(px[2] - px[12]),
        'b1_b3': np.abs(px[6] - px[16]) / 2,
        'b2_b4': np.abs(px[8] - px[18]) / 2,
        'g1_g6': np.abs(px[1] - px[11]) / 2,
        'g2_g7': np.abs(px[3] - px[13]) / 2,
        'g6_g7': np.abs(px[11] - px[13]),
        'r5_r6': np.abs(px[12] - px[14]),
        'b1_b2': np.abs(px[6] - px[8]) / 2,
        'b3_b4': np.abs(px[16] - px[18]) / 2,
        'g4_g5': np.abs(px[7] - px[9]) / 2,
        'g9_g10': np.abs(px[17] - px[19]) / 2,
        'r5_r8': np.abs(px[12] - px[22]),
        'g6_g11': np.abs(px[11] - px[21]) / 2,
        'g7_g12': np.abs(px[13] - px[23]) / 2,
        'r4_r5': np.abs(px[10] - px[12]),
        'g3_g4': np.abs(px[5] - px[7]) / 2,
        'g8_g9': np.abs(px[15] - px[17]) / 2,
        'b1_b4': np.abs(px[6] - px[18]),
        'r3_r5': np.abs(px[4] - px[12]),
        'g4_g6': np.abs(px[7] - px[11]) / 2,
        'g6_g9': np.abs(px[11] - px[17]) / 2,
        'g7_g9': np.abs(px[13] - px[17]) / 2,
        'g2_g4': np.abs(px[3] - px[7]) / 2,
        'g5_g7': np.abs(px[9] - px[13]) / 2,
        'r5_r9': np.abs(px[12] - px[24]),
        'g4_g7': np.abs(px[7] - px[13]) / 2,
        'g7_g10': np.abs(px[13] - px[19]) / 2,
        'g9_g12': np.abs(px[17] - px[23]) / 2,
        'r1_r5': np.abs(px[0] - px[12]),
        'g1_g4': np.abs(px[1] - px[7]) / 2,
        'g3_g6': np.abs(px[5] - px[11]) / 2,
        'b2_b3': np.abs(px[8] - px[16]),
        'r5_r7': np.abs(px[12] - px[20]),
        'g6_g8': np.abs(px[11] - px[15]) / 2,
        'g9_g11': np.abs(px[17] - px[21]) / 2,
    }

    # calc grads
    g_n = (difs['g4_g9'] + difs['r2_r5'] + difs['b1_b3'] +
           difs['b2_b4'] + difs['g1_g6'] + difs['g2_g7'])
    g_e = (difs['g6_g7'] + difs['r5_r6'] + difs['b1_b2'] +
           difs['b3_b4'] + difs['g4_g5'] + difs['g9_g10'])
    g_s = (difs['g4_g9'] + difs['r5_r8'] + difs['b1_b3'] +
           difs['b2_b4'] + difs['g6_g11'] + difs['g7_g12'])
    g_w = (difs['g6_g7'] + difs['r4_r5'] + difs['b1_b2'] +
           difs['b3_b4'] + difs['g3_g4'] + difs['g8_g9'])
    g_ne = (difs['b2_b3'] + difs['r3_r5'] + difs['g4_g6'] +
            difs['g7_g9'] + difs['g2_g4'] + difs['g5_g7'])
    g_se = (difs['b1_b4'] + difs['r5_r9'] + difs['g4_g7'] +
            difs['g6_g9'] + difs['g7_g10'] + difs['g9_g12'])
    g_nw = (difs['b1_b4'] + difs['r1_r5'] + difs['g4_g7'] +
            difs['g6_g9'] + difs['g1_g4'] + difs['g3_g6'])
    g_sw = (difs['b2_b3'] + difs['r5_r7'] + difs['g4_g6'] +
            difs['g7_g9'] + difs['g6_g8'] + difs['g9_g11'])

    grads = np.array([g_n, g_e, g_s, g_w, g_ne, g_se, g_nw, g_sw])
    directions = np.array(['n', 'e', 's', 'w', 'ne', 'se', 'nw', 'sw'])

    # get thresholded grads
    min_grad = grads.min()
    max_grad = grads.max()
    k1 = 1.5
    k2 = 0.5
    th = k1 * min_grad + k2 * (max_grad + min_grad)

    grads_mask = grads < th
    filterred_directions = directions[grads_mask]
    if len(filterred_directions) == 0:
        filterred_directions = directions

    # process directions
    r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
    for direction in filterred_directions:
        if direction[0] == 'n':
            if len(direction) == 1:
                r_sum += (px[12] + px[2]) / 2
                g_sum += px[7]
                b_sum += (px[6] + px[8]) / 2
            elif direction == 'ne':
                r_sum += (px[12] + px[4]) / 2
                g_sum += (px[3] + px[7] + px[9] + px[13]) / 4
                b_sum += px[8]
            else:
                r_sum += (px[12] + px[0]) / 2
                g_sum += (px[1] + px[5] + px[7] + px[11]) / 4
                b_sum += px[6]

        elif direction[0] == 's':
            if len(direction) == 1:
                r_sum += (px[12] + px[22]) / 2
                g_sum += px[17]
                b_sum += (px[16] + px[18]) / 2
            elif direction == 'se':
                r_sum += (px[12] + px[24]) / 2
                g_sum += (px[13] + px[17] + px[19] + px[23]) / 4
                b_sum += px[18]
            else:
                r_sum += (px[12] + px[20]) / 2
                g_sum += (px[11] + px[15] + px[17] + px[21]) / 4
                b_sum += px[16]

        elif direction[0] == 'e':
            r_sum += (px[12] + px[14]) / 2
            g_sum += px[13]
            b_sum += (px[8] + px[18]) / 2
        else:
            r_sum += (px[12] + px[10]) / 2
            g_sum += px[11]
            b_sum += (px[6] + px[16]) / 2

    g = np.clip(px[12] + (g_sum - r_sum) / len(filterred_directions), 0, 255)
    b = np.clip(px[12] + (b_sum - r_sum) / len(filterred_directions), 0, 255)

    # red/blue difference
    if first_blue:
        r = b
        b = px[12]
    else:
        r = px[12]

    return np.array([b, g, r], dtype=np.uint8)


def process_row(row_idx, bayer_img):
    red_row = True if row_idx % 2 == 0 else False
    res_row = np.zeros((bayer_img.shape[1] - 4, 3), dtype=np.uint8)

    for col_idx in range(res_row.shape[0]):
        px = bayer_img[row_idx:row_idx + 5, col_idx:col_idx + 5].flatten()
        px = px.astype(np.int32)

        if col_idx % 2 == 0:
            if red_row:
                pixel = calc_red_blue_center_pixel(px)
            else:
                pixel = calc_green_center_pixel(px, red_row)
        else:
            if red_row:
                pixel = calc_green_center_pixel(px, red_row)
            else:
                pixel = calc_red_blue_center_pixel(px, True)

        res_row[col_idx] = pixel

    return row_idx, res_row


def make_demosaicing(bayer_img):
    # suppose that first pixel in the image is red
    cycle_len = bayer_img.shape[0] - 4
    n_jobs = 12
    rows = Parallel(
        n_jobs=n_jobs)(delayed(process_row)(row_idx, bayer_img)
                       for row_idx in tqdm(range(cycle_len),
                                           desc='Rows processing'))

    rows.sort()
    res_img = np.array([x[1] for x in rows], dtype=np.uint8)

    return res_img


def main():
    """Application entry point."""
    args = get_args()

    # read
    bayer_img = cv2.imread(args.src_img, cv2.IMREAD_GRAYSCALE)

    # create expand img for ks=5 convolution
    bayer_img = cv2.copyMakeBorder(bayer_img, 2, 2, 2, 2, cv2.BORDER_REFLECT)

    # process
    processed_img = make_demosaicing(bayer_img)

    # save
    cv2.imwrite(args.save_to, processed_img)


if __name__ == '__main__':
    main()

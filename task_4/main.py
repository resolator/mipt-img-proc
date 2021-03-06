#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Inter-frame shift estimator."""
import cv2
import argparse

import numpy as np

from tqdm import tqdm
from pathlib import Path


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames-dir', type=Path, required=True,
                        help='Path to dir with frames for estimation.')
    parser.add_argument('--sift-features', type=int, default=400,
                        help='Number of SIFT features to detect.')
    parser.add_argument('--save-to', type=Path, required=True,
                        help='Path to save dir.')

    return parser.parse_args()


def detect_kp(img, n_features=400, verbose=False):
    # Detect
    detector = cv2.SIFT_create(nfeatures=n_features)
    kp, des = detector.detectAndCompute(img, None)

    # Show detected
    if verbose:
        img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        cv2.drawKeypoints(img, kp, img_keypoints)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img_keypoints)
        cv2.waitKey()
        cv2.destroyWindow('img')

    return kp, des


def match(des_1, des_2, k=2, top_matches=9):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_1, des_2, k=k)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    good.sort(key=lambda x: x.distance)

    return good[:top_matches]


def extract_matched_pts(kp_1, kp_2, matches):
    pts = []
    for m in matches:
        pts_1 = np.array(kp_1[m.queryIdx].pt)
        pts_2 = np.array(kp_2[m.trainIdx].pt)

        pts.append([pts_1, pts_2])

    return pts


def vote(matches, prev_kp, kp):
    dists = []
    pts = extract_matched_pts(prev_kp, kp, matches)

    for p in pts:
        dists.append(np.linalg.norm(p[0] - p[1]))

    dists = np.array(dists)
    mask = np.logical_and(dists >= np.quantile(dists, 0.2),
                          dists <= np.quantile(dists, 0.8))

    result = np.array(matches)[mask].tolist()
    if len(result) < 4:
        print('WARNING: matched less than 4 points.')

    return result[:4]


def main():
    """Application entry point."""
    args = get_args()

    args.save_to.mkdir(exist_ok=True, parents=True)

    prev_img = None
    prev_kp, prev_des = None, None
    for img_path in tqdm(sorted(args.frames_dir.glob('*.*'))):
        img = cv2.imread(str(img_path))
        kp, des = detect_kp(img, args.sift_features)

        if prev_kp is None:
            prev_img = img
            prev_kp = kp
            prev_des = des
            continue

        matches = match(prev_des, des)

        # voting
        matches = vote(matches, prev_kp, kp)
        matches = [[x] for x in matches]





        # draw
        img3 = cv2.drawMatchesKnn(
            prev_img, prev_kp, img, kp, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img3)
        cv2.waitKey()

        exit()


if __name__ == '__main__':
    main()

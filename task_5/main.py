#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Inter-frame shift estimator."""
import cv2
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames-dir', type=Path, required=True,
                        help='Path to dir with frames for estimation.')
    parser.add_argument('--save-to', type=Path,
                        help='Path image file for save plot.')

    return parser.parse_args()


def detect_kp(img, det_type='SIFT', verbose=False):
    """Detect keypoints."""
    if det_type == 'ORB':
        detector = cv2.ORB_create()
    elif det_type == 'BRISK':
        detector = cv2.BRISK_create()
    else:
        detector = cv2.SIFT_create()

    t0 = time.time()
    kp, des = detector.detectAndCompute(img, None)
    elapsed = time.time() - t0

    # Show detected
    if verbose:
        img_keypoints = np.empty((img.shape[0], img.shape[1], 3),
                                 dtype=np.uint8)
        cv2.drawKeypoints(img, kp, img_keypoints)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img_keypoints)
        cv2.waitKey()
        cv2.destroyWindow('img')

    return kp, des, elapsed / len(kp)


def match(des_1, des_2, top_matches=None):
    """Match keypoints."""
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(np.array(des_1, np.float32),
                          np.array(des_2, np.float32), k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if top_matches is None:
        return good
    else:
        good.sort(key=lambda x: x.distance)
        return good[:top_matches]


def main():
    """Application entry point."""
    args = get_args()
    if args.save_to is not None:
        args.save_to.parent.mkdir(parents=True, exist_ok=True)
        plt.title('Repeatability per frame')
        plt.xlabel('Frame number')
        plt.ylabel('Repeated %')
        plt.grid()

    det_types = ['ORB', 'SIFT', 'BRISK']
    for det_type in det_types:
        # extract keypoints
        kps, dess, elapsed = [], [], []
        for img_path in sorted(args.frames_dir.glob('*.*')):
            img = cv2.imread(str(img_path))
            kp, des, elapsed_per_point = detect_kp(img, det_type)
            kps.append(kp)
            dess.append(des)
            elapsed.append(elapsed_per_point)

        # calc repeatability
        total_frames = len(kps)
        rep = []
        rep_frames = [[] for _ in range(total_frames - 1)]
        for frame_idx in tqdm(range(len(kps)), desc=det_type + ' detector'):
            cur_kp = kps.pop(0)
            cur_des = dess.pop(0)

            cur_matches = [0] * len(cur_kp)
            # iterate over rest frames
            for sub_frame_idx, frame_dess in enumerate(dess):
                # iterate over each point in frame
                for idx, (kp, des) in enumerate(zip(cur_kp, cur_des)):
                    matches = match([des], frame_dess)
                    if len(matches) > 0:
                        cur_matches[idx] += 1

                cur_matches = np.array(cur_matches)
                rep_frames[frame_idx].extend(
                    (cur_matches / (frame_idx + 1 + sub_frame_idx)).tolist()
                )

            rep.extend((np.array(cur_matches) / total_frames).tolist())

        print(det_type,
              'time per keypoint: %f milliseconds' % (np.mean(elapsed) * 1000))
        print(det_type, 'repeatability:', np.mean(rep), end='\n\n')
        rep_frames = [np.mean(x) * 100 for x in rep_frames]

        # save plots
        if args.save_to is not None:
            plt.plot([None] + rep_frames, label=det_type)

    if args.save_to is not None:
        plt.legend()
        plt.savefig(args.save_to)


if __name__ == '__main__':
    main()

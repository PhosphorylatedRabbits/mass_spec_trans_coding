"""Crop black sides of MS1 images.
MS1 data was not recorded on the full spectra compared to MS2.
"""
import os
import plac
import logging

import matplotlib.pyplot as plt
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def crop_ms1(file_path, save_path, show=False):
    """
    save cropped image of actual aquired ms1 intensities, removing black sides,
    leaving top and bottom untouched.
    returns corners as tuple (top, left, bottom, right)
    """
    assert os.path.exists(os.path.dirname(save_path))

    imagen = cv2.imread(file_path)
    original_bottom = imagen.shape[0]
    original_top = 0

    gray = (cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY))
    thresh = (gray > 0)
    connectivity = 4

    # Perform the operation
    output = cv2.connectedComponentsWithStats(
        thresh.astype(np.uint8), connectivity
    )
    stats = output[2]

    # two biggest in case biggest is black square
    two_biggest = np.argpartition((stats[:, cv2.CC_STAT_AREA]), -2)[-2:]
    biggest = two_biggest[0]
    top = stats[biggest, cv2.CC_STAT_TOP]
    left = stats[biggest, cv2.CC_STAT_LEFT]
    right = left + stats[biggest, cv2.CC_STAT_WIDTH]
    bottom = top + stats[biggest, cv2.CC_STAT_HEIGHT]

    # if biggest reaches corners, it's black
    if left == 0 or right == imagen.shape[1]:
        biggest = two_biggest[1]
        top = stats[biggest, cv2.CC_STAT_TOP]
        left = stats[biggest, cv2.CC_STAT_LEFT]
        right = left + stats[biggest, cv2.CC_STAT_WIDTH]
        bottom = top + stats[biggest, cv2.CC_STAT_HEIGHT]

    if bottom != original_bottom or top != original_top:
        logger.warning(
            f'There is not cropped along the height: {top}-{bottom}\n'
            f'for file {file_path}'
        )

    if show:
        plt.imshow(gray, cmap='gray', interpolation='bicubic')
        plt.show()
        plt.imshow(
            gray[original_top:original_bottom, left:right],
            cmap='gray', interpolation='bicubic'
        )
        plt.show()

    cv2.imwrite(save_path, gray[original_top:original_bottom, left:right])
    return (original_top, left, original_bottom, right)


def main(raw_dir, cropped_dir):
    ms1_file_names = [
        os.path.join(raw_dir, filename) for
        filename in os.listdir(raw_dir) if
        'itms' in filename
    ]
    os.makedirs(cropped_dir, exist_ok=True)

    for filename in ms1_file_names:
        output_path = os.path.join(cropped_dir, os.path.basename(filename))
        top, left, bottom, right = crop_ms1(filename, output_path, show=False)
        print(bottom-top, right-left)


if __name__ == "__main__":
    plac.call(main)

#! /usr/bin/env python3

"""
This script will generate a video to test the ideal performance
of the seed tracker.
"""


import cv2 as cv
import numpy as np

file_name = 'test_video.avi'

seeds = [
    (200, 250),
    (0, 250),
    (-200, 250),
    (-400, 250),
    (-600, 250),
    (-800, 250),
    (-1000, 250),
    (-1200, 250)
]

if __name__ == '__main__':
    print(__doc__)

    img = np.zeros((500, 1000, 3), dtype=np.uint8)

    out = cv.VideoWriter(file_name, cv.VideoWriter_fourcc('MJPG'),
                         60.0, (1000, 500))

    while(True):
        img[::] = 0

        for i, s in enumerate(seeds):
            cv.circle(img, s, 5, (255, 255, 255), thickness=-1)
            seeds[i] = (s[0] + 5, 255)

        cv.imshow('i', img)
        out.write(img)

        key = 0xFF & cv.waitKey(1)
        if key == ord('q'):
            break

    out.release()
    cv.destroyAllWindows()

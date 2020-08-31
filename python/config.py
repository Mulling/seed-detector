import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    device_name = None
    kernel = None
    scale = None
    # how many pixels for 1cm, defaults to 10
    pixel_size = 47
    # size of fault
    fsize = 568
    # size of double
    dsize = 190
    # NOTE: all of the calculation is done pixel wise, this is only
    # used to display the data in a more redable way.

    # the direction that the video is moving
    # TODO: not fully implemented
    video_dir = None

    # the maximum distance for a tracker to associate with a
    # detection, this value MUST be scaled with sacle.
    max_dist = None
    # hsv color range
    hsv_lower = np.array([18, 25, 25])
    hsv_upper = np.array([22, 255, 255])

    roi_factor = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)

    # the function used to calculate the distance between two seeds.
    distf = None

    # the minimum amount of hits for a distance to be valid
    dist_min_hits = 30

    def is_valid(self):
        return None in (self.device_name, self.kernel,
                        self.scale, self.video_dir,
                        self.distf, self.max_dist)

    def set_max_dist(self):
        if self.scale is None:
            print('fail to calibrate tracker, None value for scale.')
        else:
            self.max_dist = 50 * (self.scale / 100.0)


# for some reason colors are represented in BGR not RGB
palette = {"black": (0, 0, 0),
           "white": (255, 255, 255),
           "red": (0, 0, 255),
           "green": (0, 255, 0),
           "blue": (255, 0, 0),
           "magenta": (255, 255, 0),
           "yellow": (0, 255, 255)
           }


def dist_horizontal(p1, p2):
    return np.abs(p1[0] - p2[0])


def dist_vertical(p1, p2):
    return np.abs(p1[1] - p2[2])

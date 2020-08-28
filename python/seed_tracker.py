#!/usr/bin/env python

import config
import cv2 as cv
import getopt
import numpy as np
import sys
import time

from sort import Sort
from simple_sort import SimpleSort


# if the tracker is in calibration mode
calibrate = False


def to_bounding_rec(c):
    x, y, w, h = cv.boundingRect(c)
    return [x, y, x + w, y + h]


def process_frame(frame, kernel_size, hsv_lower, hsv_upper):
    """
    Extract the bounding boxes from the 'frame' by first converting it
    to HSV. Blur the HSV image using a median kernel of 'kernel_size'
    and generate a binary image using the 'hsv_lower' and 'hsv_upper'
    range. Extract the contours a return its minimum enclosing
    bounding box in the form [x1, y1, x2, y2].
    """

    hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    hsv = cv.medianBlur(hsv, kernel_size)
    bimage = cv.inRange(hsv, hsv_lower, hsv_upper)

    cv.imshow('binary image', bimage)

    contours, _ = cv.findContours(bimage, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    return [to_bounding_rec(c) for c in contours]


def rescale(x, y, factor):
    """Rescale a scaled coordenate"""
    return (int(x / factor), int(y / factor))


# TODO: also needs to store the distance
def mark_dist(bboxes, frame, factor, conf, measur):
    """Draw on the frame the distace/arrow line between
    the center of each bounding and the next."""

    prev = None
    prev_id = None

    for bbox in bboxes:
        if prev is not None:
            curr = (int(((bbox[0] + bbox[2]) / 2) / factor), prev[1])
            # FIXME: this is computing the euclidian distance, should
            # be the x or y component.
            dist = conf.distf(prev, curr)

            color = None

            if dist > conf.dsize:
                color = palette["red"]
            elif dist < conf.fsize:
                color = palette["blue"]
            else:
                color = palette["green"]

            cv.arrowedLine(frame, prev, curr, color, 1)

            if (bbox[4], prev_id) not in measur:
                measur[(bbox[4], prev_id)] = np.array([dist, 1])
            else:
                # NOTE: the distance here is kept in pixels
                measur[(bbox[4], prev_id)] += [dist, 1]

            cv.putText(frame, str(bbox[4]) + '' +
                       str(round(dist / conf.pixel_size, 1)) + 'cm',
                       prev, cv.FONT_HERSHEY_PLAIN, 1, palette['yellow'])

        prev = rescale((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2,
                       factor)
        prev_id = bbox[4]


def draw_bboxes(image, bboxes, factor, color):
    """
    Draw all the bounding boxes ('bboxes') into the 'image'. 'factor'
    is is how mutch the image is scaled.

    """

    for bbox in bboxes:
        p1 = rescale(bbox[0], bbox[1], factor)
        p2 = rescale(bbox[2], bbox[3], factor)
        cv.rectangle(image, p1, p2, color, 1)


def calibrate_mode(image, conf):
    """
    TODO: doc
    """
    # image that the user sees
    image_show = image.copy()

    cv.namedWindow("calibration", cv.WINDOW_AUTOSIZE)
    # these controls are only created in calibration mode. this could
    # be done using lambdas but they only support one statement (good
    # language design)

    # lower and upper HSV limit calibration
    def hue_change_lower(x):
        conf.hsv_lower[0] = x

    def hue_change_upper(x):
        conf.hsv_upper[0] = x

    def sat_change_lower(x):
        conf.hsv_lower[1] = x

    def sat_change_upper(x):
        conf.hsv_upper[1] = x

    def val_change_lower(x):
        conf.hsv_lower[2] = x

    def val_change_upper(x):
        conf.hsv_upper[2] = x

    cv.createTrackbar("H low:\n", "calibration",
                      conf.hsv_lower[0], 180, hue_change_lower)
    cv.createTrackbar("S low:\n", "calibration",
                      conf.hsv_lower[1], 255, sat_change_lower)
    cv.createTrackbar("V low:\n", "calibration",
                      conf.hsv_lower[2], 255, val_change_lower)
    cv.createTrackbar("H up:\n", "calibration",
                      conf.hsv_upper[0], 180, hue_change_upper)
    cv.createTrackbar("S up:\n", "calibration",
                      conf.hsv_upper[1], 255, sat_change_upper)
    cv.createTrackbar("V up:\n", "calibration",
                      conf.hsv_upper[2], 255, val_change_upper)

    # scale factor calibration
    def scale_bar(x):
        conf.scale = x

    cv.createTrackbar("Scale:\n", "calibration",
                      conf.scale, 100, scale_bar)

    # kernel size calibraion
    def kernel_bar(x):
        conf.kernel = x if x & 1 != 0 else x + 1

    cv.createTrackbar("Kernel:\n", "calibration",
                      conf.kernel, 11, kernel_bar)

    # ROI(region of interest) calibraion
    def roi_up(x):
        conf.roi_factor[0] = x / 100.0

    def roi_down(x):
        conf.roi_factor[1] = np.abs(x - 100) / 100.0

    def roi_left(x):
        conf.roi_factor[2] = x / 100.0

    def roi_right(x):
        conf.roi_factor[3] = np.abs(x - 100) / 100.0

    cv.createTrackbar("Up:\n", "calibration",
                      int(conf.roi_factor[0] * 100), 70, roi_up)
    cv.createTrackbar("Down:\n", "calibration",
                      int(conf.roi_factor[1] * 100) - 100, 70, roi_down)
    cv.createTrackbar("Left:\n", "calibration",
                      int(conf.roi_factor[2] * 100), 70, roi_left)
    cv.createTrackbar("Right:\n", "calibration",
                      int(conf.roi_factor[3] * 100) - 100, 70, roi_right)

    # pixel size calibration
    def pixel_size_bar(x):
        conf.pixel_size = x

    cv.createTrackbar("Pixel size (how many pixels per 1 cm):\n",
                      "calibration", conf.pixel_size, 100, pixel_size_bar)

    # fault and double calibration
    def fault_bar(x):
        conf.fsize = x

    def double_bar(x):
        conf.dsize = x

    cv.createTrackbar("Fault size:\n", "calibration",
                      conf.fsize, 100, fault_bar)
    cv.createTrackbar("Double size:\n", "calibration",
                      conf.dsize, 400, double_bar)

    # this will only use the first frame of the video
    # NOTE: could be useful to advance frames
    bboxes = np.array([])
    while(True):
        roi = np.array([image.shape[0], image.shape[0],
                        image.shape[1], image.shape[1]], dtype=np.int16)

        roi = (roi * conf.roi_factor).astype(np.int16)

        cimage = image[roi[0]:roi[0]+roi[1], roi[2]:roi[2]+roi[3]].copy()

        cv_scale_factor = (int(cimage.shape[1] * (conf.scale / 100.0)),
                           int(cimage.shape[0] * (conf.scale / 100.0)))

        # scale the frame
        scaled_frame = cv.resize(cimage, cv_scale_factor,
                                 interpolation=cv.INTER_AREA)

        bboxes = process_frame(scaled_frame, conf.kernel,
                               conf.hsv_lower, conf.hsv_upper)
        # bounding boxes of the detections
        draw_bboxes(cimage, bboxes, float(conf.scale / 100.0),
                    palette["yellow"])

        # we copy the original frame to use for display
        image_show = image.copy()
        # this will copy the tracked image over the original image,
        # probably not efficient but it works for now
        image_show[roi[0]:roi[0]+roi[1], roi[2]:roi[2]+roi[3]] = cimage
        # draw the pixel size calibration
        cv.line(image_show, (20, 20), (20 + conf.pixel_size, 20),
                palette["green"], 2)
        cv.line(image_show, (20, 30), (20 + conf.fsize, 30),
                palette["blue"], 2)
        cv.line(image_show, (20, 40), (20 + conf.dsize, 40),
                palette["red"], 2)

        # draw a rectangle to indicate the ROI
        cv.rectangle(image_show, (roi[2], roi[0]),
                     (roi[2]+roi[3], roi[0]+roi[1]),
                     palette["black"], thickness=2)
        cv.imshow("calibration", image_show)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            sys.exit(0)
        elif key == ord("s"):
            break

    cv.destroyWindow("calibration")


def track_video_sort(conf):
    """TODO:"""

    measur = {}
    # ROI(region of interest)
    roi = np.empty((4))

    tracker = Sort()
    video = cv.VideoCapture(conf.device_name)
    tracker_counter = 1
    # map count to object tag
    tracker_map = {}

    cv_scale_factor = (0, 0)

    roi_change = True

    fps = 0
    fps_counter = 0

    start_time = time.time()

    while(video.isOpened()):
        ok, image = video.read()
        if not ok:
            # on the last frame this will cause the programm to exit
            # the results should be reported here
            break

        global calibrate

        if calibrate:
            calibrate_mode(image.copy(), conf)
            calibrate = False

        roi = np.array([image.shape[0], image.shape[0],
                        image.shape[1], image.shape[1]], dtype=np.int16)

        roi = (roi * conf.roi_factor).astype(np.int16)
        image = image[roi[0]:roi[0]+roi[1], roi[2]:roi[2]+roi[3]]

        if roi_change:
            cv_scale_factor = (int(image.shape[1] * (conf.scale / 100.0)),
                               int(image.shape[0] * (conf.scale / 100.0)))
            roi_change = False

        # scale the frame
        scaled_frame = cv.resize(image, cv_scale_factor,
                                 interpolation=cv.INTER_AREA)

        bboxes = process_frame(scaled_frame, conf.kernel,
                               conf.hsv_lower, conf.hsv_upper)

        # bounding boxes of the detections
        draw_bboxes(image, bboxes, float(conf.scale / 100.0),
                    palette["yellow"])

        tracker_results = tracker.update(np.array(bboxes))
        # sort from image going from left to right
        # TODO: add sorting when going up and down
        tracker_results = tracker_results[tracker_results[:, 0].argsort()]

        # sort does not return in sequence tag
        # fix this by binding each tag to a count value
        # tags must be sorted for this to work
        for x1, y1, x2, y2, tag in tracker_results:
            if tag not in tracker_map:
                tracker_map[tag] = tracker_counter
                tracker_counter += 1

        # bounding boxes for the tracker results
        draw_bboxes(image, tracker_results, conf.scale / 100.0,
                    palette["magenta"])

        mark_dist(tracker_results, image, conf.scale / 100.0, conf, measur)

        fps_counter += 1
        if (time.time() - start_time) > 0.250:
            fps = fps_counter / (time.time() - start_time)
            print('FPS: ', fps)
            fps_counter = 0
            start_time = time.time()

        cv.putText(image, "FPS:" + str(int(fps)), (0, 15),
                   cv.FONT_HERSHEY_PLAIN, 0.6, palette["white"])

        cv.imshow('main', image)

        key = cv.waitKey(50) & 0xFF
        if key == ord('q'):
            sys.exit(0)
        elif key == ord('p'):
            cv.waitKey(0)
        elif key == ord('r'):
            # TODO: report results and exit
            pass

    video.release()
    cv.destroyAllWindows()


def track_video(conf):
    """TODO:"""
    # store the measurment results as a tupe of (T1, T2) -> (H, D)
    # where: T1 = the id of the first detection, T2 = id of the second
    # detection, H = the number of hits, D = the distance.  NOTE: the
    # final distance is calculate as a mean of D/H. This is only done
    # at the end when showing results to avoid unessary math
    measur = {}

    # ROI(region of interest)
    roi = np.empty((4))

    tracker = SimpleSort(max_dist=conf.max_dist)
    video = cv.VideoCapture(conf.device_name)

    cv_scale_factor = (0, 0)

    roi_change = True

    fps = 0
    fps_counter = 0

    start_time = time.time()

    while(video.isOpened()):
        ok, image = video.read()
        if not ok:
            # on the last frame this will cause the program to exit
            # the results should be reported here
            break

        global calibrate

        if calibrate:
            calibrate_mode(image.copy(), conf)
            calibrate = False

        roi = np.array([image.shape[0], image.shape[0],
                        image.shape[1], image.shape[1]], dtype=np.int16)

        roi = (roi * conf.roi_factor).astype(np.int16)
        image = image[roi[0]:roi[0]+roi[1], roi[2]:roi[2]+roi[3]]

        if roi_change:
            cv_scale_factor = (int(image.shape[1] * (conf.scale / 100.0)),
                               int(image.shape[0] * (conf.scale / 100.0)))
            roi_change = False

        # scale the frame
        scaled_frame = cv.resize(image, cv_scale_factor,
                                 interpolation=cv.INTER_AREA)

        bboxes = process_frame(scaled_frame, conf.kernel,
                               conf.hsv_lower, conf.hsv_upper)

        # bounding boxes of the detections
        draw_bboxes(image, bboxes, float(conf.scale / 100.0),
                    palette["yellow"])

        tracker_results = tracker.update(np.array(bboxes))

        # sort from image going from left to right
        # TODO: add sorting when going up and down
        tracker_results = tracker_results[tracker_results[:, 0].argsort()]

        # bounding boxes for the tracker results
        draw_bboxes(image, tracker_results, conf.scale / 100.0,
                    palette["magenta"])

        mark_dist(tracker_results, image, conf.scale / 100.0, conf, measur)

        fps_counter += 1
        if (time.time() - start_time) > 0.250:
            fps = fps_counter / (time.time() - start_time)
            fps_counter = 0
            start_time = time.time()

        cv.putText(image, "FPS:" + str(int(fps)), (0, 15),
                   cv.FONT_HERSHEY_PLAIN, 0.6, palette["white"])

        cv.imshow('main', image)

        key = cv.waitKey(150) & 0xFF
        if key == ord('q'):
            sys.exit(0)
        elif key == ord('p'):
            cv.waitKey(0)
        elif key == ord('r'):
            calculate_results(measur, conf)
            pass

    video.release()
    cv.destroyAllWindows()
    calculate_results(measur, conf)


def calculate_results(measures, conf):
    if len(measures) == 0:
        return

    doubles = 0
    faults = 0
    valid_measurments = 0
    average_dist = 0
    distance_sum = 0

    measurements = []
    for v, k in measures.items():
        if k[1] >= conf.dist_min_hits:
            valid_measurments += 1
            # distance in pixels
            mean_dist = k[0] / k[1]
            measurements.append(mean_dist)
            distance_sum += mean_dist
            if mean_dist < conf.dsize:
                faults += 1
            elif mean_dist > conf.fsize:
                doubles += 1

    average_dist = distance_sum / valid_measurments

    if valid_measurments == 1:
        print("not enough measurements")
        return

    print('numbers of measurments: ', valid_measurments)
    print('prorcentage of doubles: ',
          (doubles / valid_measurments) * 100.0, '%')
    print('prorcentage of faults: ',
          (faults / valid_measurments) * 100.0, '%')
    print('average distance: ',
          round(average_dist / conf.pixel_size, 1), 'cm')

    msr = np.array(measurements)
    msr -= average_dist

    print('distribution: ', round(100.0 * (np.sqrt(np.sum(np.square(msr))
                                                   / valid_measurments - 1)
                                           / average_dist), 2), '%CV')


def usage():
    """
    Displays the usage.
    """

    print("Usage:\n"
          "seedt "
          "[-f | --file= file name] "
          "[-k | --kernel= kernel size] "
          "[-s | --scale= image scale] "
          "[-d | --dir= up down left right]")


if __name__ == '__main__':
    conf = config.Config()
    palette = config.palette

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'f:k:s:p:F:D:d:ch',
                                   ['file=', 'kernel=', 'scale=',
                                    'pixel=', 'Fault=', 'Double=',
                                    'dir='])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-f', '--file'):
            conf.device_name = arg
        elif opt in ('-k', '--kernel'):
            conf.kernel = int(arg)
        elif opt in ('-s', '--scale'):
            conf.scale = int(arg)
        elif opt in ('-p', '--pixel='):
            conf.pixel_size = int(arg)
        elif opt in ('-F', '--Fault='):
            conf.fsize = int(arg)
        elif opt in ('-D', '--Double='):
            conf.dsize = int(arg)
        elif opt in ('-d', '--dir='):
            if arg in ('up', 'down', 'left', 'right'):
                # NOTE: change this into a sorting function, no need
                # to compare string every time.
                conf.video_dir = arg
                if arg in ('up', 'down'):
                    conf.distf = config.dist_vertical
                else:
                    conf.distf = config.dist_horizontal
            else:
                print('invalid direction')
                sys.exit(1)
        elif opt == '-c':
            calibrate = True
        elif opt == '-h':
            conf.device_name = int(conf.device_name)
        else:
            print('invalid arguments')

    conf.set_max_dist()

    if conf.is_valid():
        usage()
        sys.exit(1)

    track_video(conf)
    sys.exit(0)

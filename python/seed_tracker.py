#!/usr/bin/env python

import config
import cv2 as cv
import getopt
import numpy as np
import sys
import time

# from sort import Sort
from simple_sort import SimpleSort
from simple_sort import use_centroid_model


# if the tracker is in calibration mode
calibrate = False

# out = None
# out_hsv = None
# out_bin = None


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

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # TODO: test if a Gaussian blur is better here
    hsv = cv.medianBlur(hsv, kernel_size)
    bimage = cv.inRange(hsv, hsv_lower, hsv_upper)
    bimage = cv.erode(bimage,
                      cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    # global out_hsv
    # if out_hsv is None and not calibrate:
    #     out_hsv = cv.VideoWriter('hsv.avi',
    #                              cv.VideoWriter_fourcc(*'MJPG'),
    #                              60.0, (hsv.shape[1], hsv.shape[0]))

    # global out_bin
    # if out_bin is None and not calibrate:

    #     out_bin = cv.VideoWriter('bin.avi',
    #                              cv.VideoWriter_fourcc(*'MJPG'),
    #                              60.0, (bimage.shape[1], bimage.shape[0]))

    # if not calibrate:
    #     a = np.zeros((bimage.shape[0], bimage.shape[1], 3), dtype=np.uint8)
    #     for i in range((bimage.shape[0] - 1)):
    #         for j in range((bimage.shape[1] - 1)):
    #             a[i, j, :] = bimage[i, j]

    #     out_hsv.write(hsv)
    #     out_bin.write(a)

    # cv.imshow('binary image', bimage)

    contours, _ = cv.findContours(bimage, cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)
    if use_centroid_model:
        ret = []
        for c in contours:
            x1, y1, x2, y2 = to_bounding_rec(c)
            ret.append([((x1 + x2) / 2), ((y1 + y2) / 2)])

        return ret
    else:
        return [to_bounding_rec(c) for c in contours]


def rescale(x, y, factor):
    """Rescale a scaled coordenate"""
    return (int(x / factor), int(y / factor))


def mark_dist(bboxes, frame, factor, conf, measur):
    """Draw on the frame the distace/arrow line between
    the center of each bounding and the next."""

    prev = None
    prev_id = None

    for bbox in bboxes:
        if prev is not None:
            if use_centroid_model:
                curr = (int(bbox[0] / factor), prev[1])
            else:
                curr = (int(((bbox[0] + bbox[2]) / 2) / factor), prev[1])
            dist = conf.distf(prev, curr)

            color = None

            if dist < conf.dsize:
                color = palette["blue"]
            elif dist > conf.fsize:
                color = palette["red"]
            else:
                color = palette["green"]

            cv.line(frame, prev, curr, color, 1)

            curr_id = bbox[2] if use_centroid_model else bbox[4]

            if (curr_id, prev_id) not in measur:
                measur[(curr_id, prev_id)] = np.array([dist, 1])
            else:
                # NOTE: the distance here is kept in pixels
                measur[(curr_id, prev_id)] += [dist, 1]

            cv.putText(frame, str(curr_id) + ' ' +
                       str(round(dist / conf.pixel_size, 1)) + 'cm',
                       prev, cv.FONT_HERSHEY_PLAIN, 1, palette['yellow'])

        if use_centroid_model:
            prev = rescale(bbox[0], bbox[1], factor)
            prev_id = bbox[2]
        else:
            prev = rescale((bbox[0] + bbox[2]) / 2,
                           (bbox[1] + bbox[3]) / 2,
                           factor)
            prev_id = bbox[4]


def draw_bboxes(image, bboxes, factor, color, tracker=False):
    """
    Draw all the bounding boxes ('bboxes') into the 'image'. 'factor'
    is is how mutch the image is scaled.

    """
    if use_centroid_model:
        for c in bboxes:
            if tracker:
                cv.drawMarker(image, rescale(c[0], c[1], factor), color,
                              cv.MARKER_CROSS)
            else:
                cv.circle(image, rescale(c[0], c[1], factor), 10, color)
    else:
        for bbox in bboxes:
            p1 = rescale(bbox[0], bbox[1], factor)
            p2 = rescale(bbox[2], bbox[3], factor)
            cv.rectangle(image, p1, p2, color, 1)


def calibrate_mode(image, conf, use_camera=False, cam=None):
    """
    Calibrate the tracker settings.
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
    # def kernel_bar(x):
    #     conf.kernel = x if x & 1 != 0 else x + 1

    # cv.createTrackbar("Kernel:\n", "calibration",
    #                   conf.kernel, 11, kernel_bar)

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

    def distance_bar(x):
        conf.fsize = 1.5 * x * conf.pixel_size
        conf.dsize = 0.5 * x * conf.pixel_size

    cv.createTrackbar("ideal distance in cm:",
                      "calibration", 10, 100, distance_bar)

    # fault and double calibration
    # def fault_bar(x):
    #     conf.fsize = x

    # def double_bar(x):
    #     conf.dsize = x

    # cv.createTrackbar("Fault size:\n", "calibration",
    #                   conf.fsize, 1000, fault_bar)
    # cv.createTrackbar("Double size:\n", "calibration",
    #                   conf.dsize, 1000, double_bar)

    # this will only use the first frame of the video
    # NOTE: could be useful to advance frames
    bboxes = np.array([])
    measur = {}
    tracker = SimpleSort(max_dist=conf.max_dist)

    while(True):
        if use_camera:
            ok, image = cam.read()
            image_show = image.copy()

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

        if use_camera:
            tracker_results = tracker.update(np.array(bboxes))
            # sort from image going from left to right
            # TODO: add sorting when going up and down
            tracker_results = tracker_results[tracker_results[:, 0].argsort()]
            print('tracker_results')
            # bounding boxes for the tracker results
            draw_bboxes(cimage, tracker_results, conf.scale / 100.0,
                        palette["magenta"], True)

            mark_dist(tracker_results, cimage,
                      conf.scale / 100.0, conf, measur)

        # we copy the original frame to use for display
        image_show = image.copy()
        # this will copy the tracked image over the original image,
        # probably not efficient but it works for now
        image_show[roi[0]:roi[0]+roi[1], roi[2]:roi[2]+roi[3]] = cimage
        # draw the pixel size calibration
        cv.line(image_show, (20, 20), (20 + conf.pixel_size, 20),
                palette["green"], 2)

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


def track_video(conf):
    """Track the video using simple_sort"""
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
            break

        global calibrate

        if calibrate:
            calibrate_mode(image.copy(), conf, True, video)
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
                    palette["magenta"], True)

        mark_dist(tracker_results, image, conf.scale / 100.0, conf, measur)

        fps_counter += 1
        if (time.time() - start_time) > 0.250:
            fps = fps_counter / (time.time() - start_time)
            fps_counter = 0
            start_time = time.time()

        cv.putText(image, "FPS:" + str(int(fps)), (0, 15),
                   cv.FONT_HERSHEY_PLAIN, 0.6, palette["white"])

        # global out
        # if out is None:
        #     out = cv.VideoWriter('tracking.avi',
        #                          cv.VideoWriter_fourcc(*'MJPG'),
        #                          60.0, (image.shape[1], image.shape[0]))
        # out.write(image)

        cv.imshow('main', image)

        key = cv.waitKey(1) & 0xFF
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

    # out.release()
    # out_hsv.release()
    # out_bin.release()


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
                doubles += 1
            elif mean_dist > conf.fsize:
                faults += 1

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

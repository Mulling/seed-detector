import os
import cv2 as cv
import numpy as np


def toggle_focus(focus, device='/dev/video2'):
    """toggle camera focus for the v4l2 driver"""

    command = 'v4l2-ctl -d' + device + ' -c focus_auto=' + str(int(focus))
    print(command)
    os.system(command)


def calibrate_camera(device=2, board_size=(8, 6)):
    """calibrate the camera using a chessboard"""
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)

    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    video = cv.VideoCapture(device)

    obj_p = []
    img_p = []

    while(video.isOpened()):
        ok, frame = video.read()

        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

        ret, cor = cv.findChessboardCorners(gray, board_size, cv.CALIB_CB_ADAPTIVE_THRESH)

        if ret is True:
            print("Found chessboard")
            obj_p.append(objp)

            ncor = cv.cornerSubPix(gray, cor, (11, 11), (-1, -1), criteria)
            img_p.append(ncor)
            nframe = cv.drawChessboardCorners(frame, board_size, ncor, ret)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_p, img_p, gray.shape[::-1], None, None)

            h, w = frame.shape[:2]
            ncm, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            print("camera matrix")
            print(ncm)

            print("distance")
            print(dist)

            print("distortion matrix")
            print(mtx)

            dst = cv.undistort(frame, mtx, dist, None, ncm)

            # x, y, w, h = roi
            # dst = dst[y:y+h, x:x+w]
            cv.imshow('calibrated', dst)
            cv.imshow('uncalibrated', nframe)

            with open("calibration.txt", "w") as file:
                file.write(str(ncm) + '\n')
                file.write(str(dist) + '\n')
                file.write(str(mtx) + '\n')

            break

    while(True):
        if cv.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    calibrate_camera()

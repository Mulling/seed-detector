package com.example.seedtracker;

import org.jetbrains.annotations.NotNull;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.video.KalmanFilter;

final public class Tracker {

    KalmanFilter kf;

    static int count = 0;

    public int age;
    public int hit;
    public int id;

    Tracker(Rect bBox){
        kf = new KalmanFilter(6, 4,0);

        Mat transitionMat = new Mat(6, 6, CvType.CV_32F);
        float[] tMat = {
                1, 0, 0, 0, 1, 0,
                0, 1, 0, 0, 0, 1,
                0, 0, 1, 0, 0, 0,
                0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 1
        };
        transitionMat.put(0,0, tMat);
        kf.set_transitionMatrix(transitionMat);

        Mat measurementMat = new Mat(4, 6, CvType.CV_32F);
        float[] mMat = {
                1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0,
                0, 0, 0, 1, 0, 0
        };
        measurementMat.put(0,0, mMat);
        kf.set_measurementMatrix(measurementMat);

        Mat noiseCovMat = new Mat(6,6, CvType.CV_32F);
        float[] nMat = {
                10,  0,  0,  0,     0,     0,
                 0, 10,  0,  0,     0,     0,
                 0,  0, 10,  0,     0,     0,
                 0,  0,  0, 10,     0,     0,
                 0,  0,  0,  0, 10000,     0,
                 0,  0,  0,  0,     0, 10000
        };
        noiseCovMat.put(0,0, nMat);
        kf.set_processNoiseCov(noiseCovMat);

        Mat statePostMat = new Mat(6, 1, CvType.CV_32F);
        bBoxToArr(bBox, statePostMat);

        float[] aux = new float[6];
        statePostMat.get(0,0, aux);
//        this is the OOP police you cannot access the content's of a class directly, no matter how inconvenient.
        aux[4] = 0.0f;
        aux[5] = 0.0f;
        statePostMat.put(0,0, aux);

        kf.set_statePost(statePostMat);

        age = 0;
        hit = 0;

        id = count;
        count++;

//        Log.i("OpenCV::seedTracker", "Tracker created with ID: " + id
//                + "\n" +
//                kf.get_transitionMatrix().dump() + "\n" +
//                kf.get_measurementMatrix().dump() + "\n" +
//                kf.get_processNoiseCov().dump() + "\n" +
//                kf.get_statePost().dump()
//        );
    }

    public void update(Rect bBox){
        age = 0;
        hit++;
        Mat arr = new Mat(4, 1, CvType.CV_32F);
        bBoxToArr(bBox, arr);
        kf.correct(arr);
        arr.release();
    }

    public Rect predict(){
        age++;
        Rect bBox = new Rect();

        Mat arr = kf.predict();
        arrToBBox(bBox, arr);
        arr.release();
        return bBox;
    }

    public static void bBoxToArr(Rect bBox, @NotNull Mat dst){
//        NOTE: this could be done better, but for now it will work
//        double -> float

        float x = (float) (bBox.x + (bBox.width / 2.0));
        float y = (float) (bBox.y + (bBox.height / 2.0));
        float[] aux = {x, y, (float) bBox.area(), (float)bBox.width / (float)bBox.height};
        dst.put(0,0, aux);
//        Log.i("OpenCV::seedTracker", "bBoxToArr: " + bBox.toString() + " to ->" + dst.dump());
    }

    public static void arrToBBox(Rect dst, @NotNull Mat arr){
//        float -> double
        float[] arrM = new float[6];
        arr.get(0, 0, arrM);
        float w = (float) Math.sqrt(arrM[3] * arrM[2]);
        float h = arrM[2] / w;

        double [] box = new double[4];

        box[0] = arrM[0] - (w / 2.0);
        box[1] = arrM[1] - (h / 2.0);
        box[2] = w;
        box[3] = h;
        dst.set(box);
//        Log.i("OpenCV::seedTracker", "arrToBBox: " + arr.dump() + " to ->" +  dst.toString());
    }
}
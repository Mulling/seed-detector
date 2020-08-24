package com.example.seedtracker;

import org.opencv.core.Scalar;

final public class Config {
    public static int kernel = 3;
    public static int scale = 25;
    public static int pixelSize = 12;
    public static int fSize = 80;
    public static int dSize = 30;
    public static float max_dist = 50 * (scale / 100.0f);
    public static Scalar hsvLower = new Scalar(25, 100,  20);
    public static Scalar hsvUpper = new Scalar(43, 255, 255);
    public static float[] roiFactor = new float[]{0.0f, 1.0f, 0.0f, 1.0f};
    public static int distMinHits = 30;

    public void setMaxDist(){
        max_dist = 100 * (scale / 100.0f);
    }
}

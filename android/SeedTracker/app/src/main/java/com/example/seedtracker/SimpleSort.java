package com.example.seedtracker;

import android.util.Log;
import android.util.Pair;

import org.opencv.core.Rect;

import java.util.ArrayList;
import java.util.Arrays;

import static java.lang.Math.max;

final public class SimpleSort {

    private static ArrayList<Tracker> trackers;

    private static int maxAge;
    private static int minHits;
    private static double maxDist;

    private static ArrayList<Rect> rTrks = new ArrayList<>();
    private static ArrayList<Integer> toDel = new ArrayList<>();

    private static ArrayList<Pair<Integer, Integer>> matches = new ArrayList<>();
    private static ArrayList<Integer> unmatchedDets = new ArrayList<>();

    SimpleSort(int maxAge, int minHits, double maxDist){
        trackers = new ArrayList<>();
        SimpleSort.maxAge = maxAge;
        SimpleSort.minHits = minHits;
        SimpleSort.maxDist = maxDist;
    }

    public static ArrayList<Pair<Rect, Integer>> update(ArrayList<Rect> dets){
        ArrayList<Pair<Rect, Integer>> ret = new ArrayList<>();
        rTrks.clear();
        toDel.clear();
        matches.clear();
        unmatchedDets.clear();
        for(int i = 0; i < trackers.size(); i++){
            rTrks.add(new Rect());
        }

        for(int i = 0; i < trackers.size(); i++){
            Rect det = trackers.get(i).predict();
//            Log.i("OpenCV::seedTracker", "Tracker prediction: " + det.toString());
            rTrks.set(i, det);
            if(Double.isNaN(det.x) || Double.isNaN(det.y) || Double.isNaN(det.height) || Double.isNaN(det.width)){
                toDel.add(i);
            }
        }

        for(int i = toDel.size() - 1; i >= 0; i--){
            trackers.remove(i);
            rTrks.remove(i);
        }

        associate(rTrks, dets, matches, unmatchedDets);

        for(int i = 0; i < trackers.size(); i++){
            for(Pair<Integer, Integer> m : matches){
                if(m.second == i){
                    Rect det = dets.get(m.first);
                    trackers.get(i).update(det);
                    if(trackers.get(i).hit > minHits){
                        ret.add(new Pair<>(det, trackers.get(i).id));
                    }
                }
            }
        }
//        instantiate new trackers
        for(Integer i : unmatchedDets){
            trackers.add(new Tracker(dets.get(i)));
        }
//        remove old trackers
        for(int i = trackers.size() - 1; i >= 0; i--){
            if(trackers.get(i).age > maxAge){
                Log.i("OpenCV::seedTracker", "Removed tracker with ID: " + trackers.get(i).id);
                trackers.remove(i);
            }
        }
        return ret;
    }

    public static void associate(ArrayList<Rect> trks, ArrayList<Rect> dets, ArrayList<Pair<Integer, Integer>> dstMatches, ArrayList<Integer> dstUnmatchedDets){
        if(trks.size() == 0) {
            for (int i = 0; i < dets.size(); i++) {
                dstUnmatchedDets.add(i);
            }
            return;
        }

        int dim = max(trks.size(), dets.size());

//        the dimensions must be equal.
        float[][] mat = new float[dim][dim];
        for(float[] m : mat) Arrays.fill(m, 5000);

        for(int i = 0; i < dets.size(); i++){
            for(int j = 0; j < trks.size(); j++){
                mat[i][j] = euclid(trks.get(j), dets.get(i));
            }
        }

//        this is only a temporary solution
//        FIXME: implement the associate and linear assignment in C++ and use the native runtime
        LinearAssignment la = new LinearAssignment(mat);
//        the return is in the form {{d, t}...}
        int[][] matches = la.findOptimalAssignment();

//        remove padding matches and
//        filter detections with high distance
        ArrayList<int[]> matchesList = new ArrayList<>();
        for(int[] m : matches){
            int id = m[0];
            int it = m[1];
            if(id < dets.size() && it < trks.size()){
                matchesList.add(m);
                if(mat[id][it] < maxDist){
                    Log.i("OpenCV::seedTracker", "Tracker: " + trackers.get(it).id + " matched " + " Detection: " + id + " distance = " + euclid(trks.get(it), dets.get(id)) );
                    dstMatches.add(new Pair<>(id, it));
                }
            }
        }

//      add unmatched detections
        for(int i = 0; i < dets.size(); i++){
            boolean found = false;
            for (int[] match : matchesList) {
                if (i == match[0]) {
                    found = true;
                    break;
                }
            }
            if(!found)
                unmatchedDets.add(i);
        }
    }

    private static float euclid(Rect p1, Rect p2){
//        p1 and p2 are bounding boxes
        float dx = (p1.x + (p1.width / 2.0f)) - (p2.x + (p2.width / 2.0f));
        float dy = (p1.y + (p1.height / 2.0f)) - (p2.y + (p2.height / 2.0f));

        return (float) Math.sqrt(dx * dx + dy * dy);
    }
}

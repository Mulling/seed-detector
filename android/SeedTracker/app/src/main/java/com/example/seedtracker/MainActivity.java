package com.example.seedtracker;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Pair;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_PLAIN;
import static org.opencv.imgproc.Imgproc.RETR_LIST;
import static org.opencv.imgproc.Imgproc.putText;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private TextView hsvLowerID;
    private TextView hsvUpperID;

    private CameraBridgeViewBase cameraBridgeViewBase;
    private BaseLoaderCallback baseLoaderCallback;
    private Mat img;
    private Mat sImg;
    private Mat bin;
    private Mat hsvBlur;
    private Mat hsv;
    private Mat h;

    private boolean mode = false;
    private boolean gatheringData = false;

    private Config conf;

    private Map<Pair<Integer, Integer>, Pair<Integer, Integer>> measurements = new HashMap<>();

    private SimpleSort ss = new SimpleSort(3, 5, 25);

    private AlertDialog.Builder resultsPopUp;

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        hsvLowerID = findViewById(R.id.hsvLowerID);
        hsvUpperID = findViewById(R.id.hsvUpperID);

        resultsPopUp = new AlertDialog.Builder(this)
                .setTitle("Results")
                .setMessage("No results").setOnDismissListener(new DialogInterface.OnDismissListener() {
                    @Override
                    public void onDismiss(DialogInterface dialogInterface) {
                        resultsPopUp.setMessage("No Results.");
                        measurements.clear();
                    }
                })
                .setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                        resultsPopUp.setMessage("No Results.");
                        Toast.makeText(MainActivity.this, "Resetting results.", Toast.LENGTH_SHORT).show();
                        measurements.clear();
                    }
                });

        hsvLowerID.setText("Low(H:" +  Config.hsvLower.val[0] + " S:" + Config.hsvLower.val[1] + " V:" + Config.hsvLower.val[2] + ")");
        hsvUpperID.setText("Up(H:" +  Config.hsvUpper.val[0] + " S:" + Config.hsvUpper.val[1] + " V:" + Config.hsvUpper.val[2] + ")");

        Button startButton = findViewById(R.id.startButton);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Toast.makeText(MainActivity.this, "Started.", Toast.LENGTH_SHORT).show();
                gatheringData = true;
            }
        });

        Button resultButton = findViewById(R.id.resultsButton);
        resultButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                gatheringData = false;
                resultsPopUp.setMessage(calculateResults());
                resultsPopUp.show();
            }
        });

        Button clearButton = findViewById(R.id.clearButton);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Toast.makeText(MainActivity.this, "Measurements cleared.", Toast.LENGTH_SHORT).show();
                gatheringData = false;
                measurements.clear();
                resultsPopUp.setMessage("No results.");

            }
        });

        SeekBar hLowerSeekBar = findViewById(R.id.hLower);
        hLowerSeekBar.setProgress((int) Config.hsvLower.val[0]);
        hLowerSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Config.hsvLower.val[0] = i;
                hsvLowerID.setText("Low(H:" +  Config.hsvLower.val[0] + " S:" + Config.hsvLower.val[1] + " V:" + Config.hsvLower.val[2] + ")");
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        SeekBar sLowerSeekBar = findViewById(R.id.sLower);
        sLowerSeekBar.setProgress((int) Config.hsvLower.val[1]);
        sLowerSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Config.hsvLower.val[1] = i;
                hsvLowerID.setText("Low(H:" +  Config.hsvLower.val[0] + " S:" + Config.hsvLower.val[1] + " V:" + Config.hsvLower.val[2] + ")");
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        SeekBar vLowerSeekBar = findViewById(R.id.vLower);
        vLowerSeekBar.setProgress((int) Config.hsvLower.val[2]);
        vLowerSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Config.hsvLower.val[2] = i;
                hsvLowerID.setText("Low(H:" +  Config.hsvLower.val[0] + " S:" + Config.hsvLower.val[1] + " V:" + Config.hsvLower.val[2] + ")");
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        SeekBar hUpperSeekBar = findViewById(R.id.hUpper);
        hUpperSeekBar.setProgress((int) Config.hsvUpper.val[0]);
        hUpperSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Config.hsvUpper.val[0] = i;
                hsvUpperID.setText("Up(H:" +  Config.hsvUpper.val[0] + " S:" + Config.hsvUpper.val[1] + " V:" + Config.hsvUpper.val[2] + ")");
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        SeekBar sUpperSeekBar = findViewById(R.id.sUpper);
        sUpperSeekBar.setProgress((int) Config.hsvUpper.val[1]);
        sUpperSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Config.hsvUpper.val[1] = i;
                hsvUpperID.setText("Up(H:" +  Config.hsvUpper.val[0] + " S:" + Config.hsvUpper.val[1] + " V:" + Config.hsvUpper.val[2] + ")");
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        SeekBar vUpperSeekBar = findViewById(R.id.vUpper);
        vUpperSeekBar.setProgress((int) Config.hsvUpper.val[2]);
        vUpperSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Config.hsvUpper.val[2] = i;
                hsvUpperID.setText("Up(H:" +  Config.hsvUpper.val[0] + " S:" + Config.hsvUpper.val[1] + " V:" + Config.hsvUpper.val[2] + ")");
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        SeekBar doubleSeekBar = findViewById(R.id.douleSize);
        doubleSeekBar.setProgress(Config.dSize);
        doubleSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Config.dSize = i;
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        SeekBar faultSeekBar = findViewById(R.id.faultSize);
        faultSeekBar.setProgress(Config.fSize);
        faultSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                Config.fSize = i;
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        SeekBar centimeterBar = findViewById(R.id.centimeter);
        centimeterBar.setProgress(Config.pixelSize);
        centimeterBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int i, boolean b) {
                if(i == 0){
                    Config.pixelSize = 1;
                } else {
                    Config.pixelSize = i;
                }
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });

//        if(OpenCVLoader.initDebug()){
//            Toast.makeText(getApplicationContext(), "OpenCV loaded.", Toast.LENGTH_SHORT).show();
//        } else {
//            Toast.makeText(getApplicationContext(), "OpenCV failed to load.", Toast.LENGTH_SHORT).show();
//        }

        ActivityCompat.requestPermissions(MainActivity.this, new String[] {Manifest.permission.CAMERA}, 1);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.camView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCameraPermissionGranted();
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.setOnTouchListener(new View.OnTouchListener() {
            @SuppressLint("ClickableViewAccessibility")
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                if(!mode) {
                    findViewById(R.id.toolBar).setVisibility(View.GONE);
                    findViewById(R.id.buttonPanel).setVisibility(View.VISIBLE);
                    Toast.makeText(getApplicationContext(), "Record", Toast.LENGTH_SHORT).show();
                } else {
                    findViewById((R.id.toolBar)).setVisibility(View.VISIBLE);
                    findViewById(R.id.buttonPanel).setVisibility(View.GONE);
                    Toast.makeText(getApplicationContext(), "Settings", Toast.LENGTH_SHORT).show();
                }
                mode = !mode;
                return false;
            }
        });

        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                if (status == LoaderCallbackInterface.SUCCESS) {
                    cameraBridgeViewBase.enableView();
                } else {
                    super.onManagerConnected(status);
                }
            }
        };
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {// If request is cancelled, the result arrays are empty.
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                cameraBridgeViewBase.setCameraPermissionGranted();
            } else {
                this.finishAffinity();
            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        img = inputFrame.rgba();

        Imgproc.resize(img, sImg, new Size(
                img.cols() * (Config.scale / 100.0),
                img.rows() * (Config.scale / 100.0))
        );

        ArrayList<Rect> bBoxes = processFrame(sImg, Config.kernel, Config.hsvLower, Config.hsvUpper);
        drawBBoxes(img, bBoxes, new Scalar(0, 255, 0, 255));

        ArrayList<Pair<Rect, Integer>> tRes = SimpleSort.update(bBoxes);
        drawTracked(img, tRes);

        if(!mode){
            drawSizeCalibrationLines(img);
        }
        markDistance(img, tRes);

        return img;
    }

    @Override
    public void onCameraViewStopped() {
        img.release();
        sImg.release();
        hsv.release();
        h.release();
        hsvBlur.release();
        bin.release();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        img = new Mat(height, width, CvType.CV_8UC4);
        hsv = new Mat(img.rows(), img.cols(), CvType.CV_8UC3);
        hsvBlur = new Mat(img.rows(), img.cols(), CvType.CV_8UC3);
        bin = new Mat(img.rows(), img.cols(), CvType.CV_8UC3);
        sImg = new Mat();
        h = new Mat();
        measurements.clear();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(), "Fail to load OpenCV.", Toast.LENGTH_SHORT).show();
        } else {
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }

    protected void drawBBoxes(Mat img, ArrayList<Rect> bBoxes, Scalar color){
        for(Rect r : bBoxes){
            Imgproc.rectangle(img, rescale(r.br()), rescale(r.tl()), color, 1);
        }
    }

    @SuppressLint("DefaultLocale")
    protected String calculateResults() {
        if (measurements.isEmpty()) {
            return "No results.";
        }
        if (measurements.size() == 1) {
            return "Not enough measurements.";
        }

        int dub = 0;
        int fault = 0;
        int valid = 0;
        double averageDist;
        double distanceSum = 0;

        ArrayList<Integer> validMeasurements = new ArrayList<>();

        for (Map.Entry<Pair<Integer, Integer>, Pair<Integer, Integer>> kv : measurements.entrySet()) {
            Pair<Integer, Integer> k = kv.getKey();
            Pair<Integer, Integer> v = kv.getValue();
            if (v.second > Config.distMinHits) {
                valid++;
                int meanDist = v.first / v.second;
                validMeasurements.add(meanDist);
                distanceSum += meanDist;
                if (meanDist < Config.dSize) {
                    dub++;
                } else if (meanDist > Config.fSize) {
                    fault++;
                }
            }
        }
        if (valid == 1) {
            return "Not enough measurements.";
        }
        averageDist = distanceSum / (double) valid;

        double sigma = 0;
        for (Integer i : validMeasurements) {
            sigma += Math.pow(i - averageDist, 2);
        }

        sigma = Math.sqrt(sigma / (valid - 1));
        double CV = 100 * sigma / averageDist;
        return String.format("Number of measurements = %d\nDoubles = %.2f %%\nFaults = %.2f %%\nC.V = %.2f %%\nAverage distance = %.2fcm", valid, (float)(dub / valid) * 100.0, (float)(fault / valid) * 100, (float)CV, (float)averageDist / Config.pixelSize);
    }

    protected void drawTracked(Mat img, ArrayList<Pair<Rect, Integer>> bBoxes){
        for(Pair<Rect, Integer> r : bBoxes){
            putText(img, r.second.toString(), rescale(r.first.br()), FONT_HERSHEY_PLAIN, 1, new Scalar(0, 255, 255, 255));
            Imgproc.drawMarker(img, rescale(new Point(r.first.x + (r.first.width / 2.0), r.first.y + (r.first.height / 2.0))), new Scalar(0, 255, 255, 255));
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    protected void markDistance(Mat img, ArrayList<Pair<Rect, Integer>> bBoxes){
        bBoxes.sort(new Comparator<Pair<Rect, Integer>>() {
            @Override
            public int compare(Pair<Rect, Integer> p1, Pair<Rect, Integer> p2) {
                return p1.first.x - p2.first.x;
            }
        });

        if(bBoxes.isEmpty()){
            return;
        }

        Pair<Point, Integer> prev = new Pair<>(new Point(), -1);
        for(Pair<Rect, Integer> p : bBoxes){
            Pair<Point, Integer> curr = new Pair<>(rescale(new Point(p.first.x + (p.first.width / 2.0), p.first.y + (p.first.height / 2.0))), p.second);

            if(prev.second != -1){
                int dist = (int) Math.abs(curr.first.x - prev.first.x);

                if(gatheringData){
                    //                very good language BTW
                    Pair<Integer, Integer> k = new Pair<>(curr.second, prev.second);
                    if(!measurements.containsKey(k)){
                        measurements.put(k, new Pair<>(dist, 1));
                    } else {
                        Pair<Integer, Integer> v = measurements.get(k);
                        assert v != null;
                        measurements.put(k, new Pair<>(dist + v.first, v.second + 1));
                    }
                }
                Imgproc.putText(img, String.valueOf(Math.round((double)dist / Config.pixelSize)), new Point(prev.first.x + (dist / 2.0), prev.first.y - 10), FONT_HERSHEY_PLAIN, 1, new Scalar(0,255,128,255));
                Imgproc.line(img, prev.first, new Point(prev.first.x + dist, prev.first.y), new Scalar(0, 128, 128, 255), 1);

            }
            prev = curr;
        }
    }

    protected void drawSizeCalibrationLines(Mat img){
        Imgproc.line(img,
                new Point((img.cols() / 1.2), (img.rows() / 2.0) + 30),
                new Point((img.cols() / 1.2) - Config.dSize, (img.rows() / 2.0) + 30),
                new Scalar(0, 0, 255, 255), 1);
        Imgproc.line(img,
                new Point((img.cols() / 1.2), (img.rows() / 2.0) - 30),
                new Point((img.cols() / 1.2) - Config.fSize, (img.rows() / 2.0) - 30),
                new Scalar(255, 0, 0, 255), 1);
        Imgproc.line(img,
                new Point((img.cols() / 1.2), (img.rows() / 2.0)),
                new Point((img.cols() / 1.2) - Config.pixelSize, (img.rows() / 2.0)),
                new Scalar(0, 255, 0, 255), 1);

    }

    protected ArrayList<Rect> processFrame(Mat img, int kernel, Scalar hsvLower, Scalar hsvUpper){
        ArrayList<MatOfPoint> contour = new ArrayList<>();
        Mat h = new Mat();

        ArrayList<Rect> ret = new ArrayList<>();

        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_RGB2HSV);
        Imgproc.medianBlur(hsv, hsvBlur, kernel);
        Core.inRange(hsvBlur, hsvLower, hsvUpper, bin);

        Imgproc.findContours(bin, contour, h, RETR_LIST, CHAIN_APPROX_SIMPLE);

        for(MatOfPoint m : contour){
            ret.add(Imgproc.boundingRect(m));
        }

        return ret;
    }

    protected Point rescale(Point p){
        return new Point(p.x / (Config.scale / 100.0), p.y / (Config.scale / 100.0));
    }
}
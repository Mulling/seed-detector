package com.example.seedtracker;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.util.Log;
import android.util.Pair;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.NumberPicker;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.skydoves.colorpickerview.ActionMode;
import com.skydoves.colorpickerview.ColorEnvelope;
import com.skydoves.colorpickerview.ColorPickerDialog;
import com.skydoves.colorpickerview.ColorPickerView;
import com.skydoves.colorpickerview.listeners.ColorEnvelopeListener;
import com.skydoves.colorpickerview.listeners.ColorListener;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
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

import static java.lang.Math.floor;
import static java.lang.Math.round;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2HSV;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_PLAIN;
import static org.opencv.imgproc.Imgproc.MORPH_ELLIPSE;
import static org.opencv.imgproc.Imgproc.MORPH_OPEN;
import static org.opencv.imgproc.Imgproc.RETR_EXTERNAL;
import static org.opencv.imgproc.Imgproc.RETR_LIST;
import static org.opencv.imgproc.Imgproc.getStructuringElement;
import static org.opencv.imgproc.Imgproc.putText;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
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

    private SimpleSort ss = new SimpleSort(3, 5, Config.max_dist);

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        ActivityCompat.requestPermissions(MainActivity.this, new String[] {Manifest.permission.CAMERA}, 1);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.camView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCameraPermissionGranted();
        cameraBridgeViewBase.setCvCameraViewListener(this);

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

        final AlertDialog.Builder resultsPopUp = new AlertDialog.Builder(this)
                .setTitle("Results")
                .setMessage("No results").setOnDismissListener(new DialogInterface.OnDismissListener() {
                    @Override
                    public void onDismiss(DialogInterface dialogInterface) {
                        measurements.clear();
                    }
                })
                .setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                        measurements.clear();
                    }
                });

        final Dialog colorCalDialog = new Dialog(MainActivity.this);
        colorCalDialog.setContentView(R.layout.color_dialog);

        final Dialog distanceDialog = new Dialog(MainActivity.this);
        distanceDialog.setContentView(R.layout.distance_dialog);

        Button distanceButton = findViewById(R.id.distanceButton);
        distanceButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                distanceDialog.show();
            }
        });

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

        ImageButton colorCalButton = findViewById(R.id.colCal);
        colorCalButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                colorCalDialog.show();
            }
        });

        NumberPicker.Formatter formatter = new NumberPicker.Formatter() {
            @NotNull
            @Contract(pure = true)
            @Override
            public String format(int i) {
                return i+"cm";
            }
        };

        final NumberPicker idealNP = distanceDialog.findViewById(R.id.idealDistance);
        idealNP.setFormatter(formatter);
        idealNP.setMaxValue(50);
        idealNP.setMinValue(0);

        final NumberPicker faultNP = distanceDialog.findViewById(R.id.faultDistance);
        faultNP.setFormatter(formatter);
        faultNP.setMaxValue(75);
        faultNP.setMinValue(0);

        final NumberPicker doubleNP = distanceDialog.findViewById(R.id.doubleDistance);
        doubleNP.setFormatter(formatter);
        doubleNP.setMaxValue(25);
        doubleNP.setMinValue(0);

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
                Config.dSize = doubleNP.getValue() * Config.pixelSize;
                Config.fSize = faultNP.getValue() * Config.pixelSize;
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        
        idealNP.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
//                HOOKS
                int v = numberPicker.getValue();
                int di = round(v * 0.5f);
                int fi = round(v * 1.5f);

                doubleNP.setValue(di);
                Config.dSize = di * Config.pixelSize;

                faultNP.setValue(fi);
                Config.fSize = fi * Config.pixelSize;
            }
        });

        doubleNP.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                Config.dSize = i * Config.pixelSize;
            }
        });

        faultNP.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                Config.fSize = i * Config.pixelSize;
            }
        });

        final NumberPicker lowH = colorCalDialog.findViewById(R.id.lowH);
        lowH.setMinValue(0); lowH.setMaxValue(180);
        lowH.setValue((int)Config.hsvLower.val[0]);
        lowH.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                Config.hsvLower.val[0] = (float)i;
            }
        });

        final NumberPicker lowS = colorCalDialog.findViewById(R.id.lowS);
        lowS.setMaxValue(0); lowS.setMaxValue(255);
        lowS.setValue((int)Config.hsvLower.val[1]);
        lowS.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                Config.hsvLower.val[1] = (float)i;
            }
        });

        final NumberPicker lowV = colorCalDialog.findViewById(R.id.lowV);
        lowV.setMinValue(0); lowV.setMaxValue(255);
        lowV.setValue((int)Config.hsvLower.val[2]);
        lowV.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                Config.hsvLower.val[2] = (float)i;
            }
        });

        final NumberPicker upH = colorCalDialog.findViewById(R.id.upH);
        upH.setMinValue(0); upH.setMaxValue(180);
        upH.setValue((int)Config.hsvUpper.val[0]);
        upH.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                Config.hsvUpper.val[0] = (float)i;
            }
        });

        final NumberPicker upS = colorCalDialog.findViewById(R.id.upS);
        upS.setMinValue(0); upS.setMaxValue(255);
        upS.setValue((int)Config.hsvUpper.val[1]);
        upS.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                Config.hsvUpper.val[1] = (float)i;
            }
        });

        final NumberPicker upV = colorCalDialog.findViewById(R.id.upV);
        upV.setMinValue(0); upV.setMaxValue(255);
        upV.setValue((int)Config.hsvUpper.val[2]);
        upV.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
            @Override
            public void onValueChange(NumberPicker numberPicker, int i, int i1) {
                Config.hsvUpper.val[2] = (float)i;
            }
        });

        final AlertDialog colorPicker = new ColorPickerDialog.Builder(this)
                .setTitle("Seed Color")
                .setPreferenceName("MyColorPickerDialog")
                .setPositiveButton("OK",
                        new ColorEnvelopeListener() {
                            @Override
                            public void onColorSelected(ColorEnvelope envelope, boolean fromUser) {
                                int [] color = envelope.getArgb();
                                Mat col = new Mat(1, 1, CvType.CV_8UC3);
                                Mat colHSV = new Mat();
                                col.setTo(new Scalar(color[1], color[2], color[3]));

                                Imgproc.cvtColor(col, colHSV, COLOR_RGB2HSV);

                                Config.setHSVRange((int)colHSV.get(0,0)[0]);

//                                FIXME: HOOKS, WHERE ARE THE HOOKS

                                lowH.setValue((int)Config.hsvLower.val[0]);
                                lowS.setValue((int)Config.hsvLower.val[1]);
                                lowV.setValue((int)Config.hsvLower.val[2]);

                                upH.setValue((int)Config.hsvUpper.val[0]);
                                upS.setValue((int)Config.hsvUpper.val[1]);
                                upV.setValue((int)Config.hsvUpper.val[2]);

                                colHSV.release();
                                col.release();
                            }
                        })
                .setNegativeButton("Cancel",
                        new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialogInterface, int i) {
                                dialogInterface.dismiss();
                            }
                        })
                .attachAlphaSlideBar(false)
                .attachBrightnessSlideBar(true)
                .setBottomSpace(12)
                .create();

        Button colorButton = findViewById(R.id.colorButton);
        colorButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                colorPicker.show();
            }
        });
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

    protected void drawBBoxes(Mat img, @NotNull ArrayList<Rect> bBoxes, Scalar color){
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
        if (valid <= 1) {
            return "Not enough valid measurements.";
        }
        averageDist = distanceSum / (double) valid;

        double sigma = 0;
        for (Integer i : validMeasurements) {
            sigma += Math.pow(i - averageDist, 2);
        }

        sigma = Math.sqrt(sigma / (valid - 1));
        double CV = 100 * sigma / averageDist;
        return String.format("Number of measurements = %d\nDoubles = %.2f %%\nFaults = %.2f %%\nC.V = %.2f %%\nAverage distance = %.2fcm",
                valid,
                (float)(dub / (float)valid) * 100.0,
                (float)(fault / (float)valid) * 100,
                (float)CV,
                (float)averageDist / Config.pixelSize);
    }

    protected void drawTracked(Mat img, @NotNull ArrayList<Pair<Rect, Integer>> bBoxes){
        for(Pair<Rect, Integer> r : bBoxes){
            putText(img, r.second.toString(), rescale(r.first.br()), FONT_HERSHEY_PLAIN, 1, new Scalar(0, 255, 255, 255));
            Imgproc.drawMarker(img, rescale(new Point(r.first.x + (r.first.width / 2.0), r.first.y + (r.first.height / 2.0))), new Scalar(0, 255, 255, 255));
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    protected void markDistance(Mat img, @NotNull ArrayList<Pair<Rect, Integer>> bBoxes){
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
//                FIXME:
                if(dist < Config.dSize){
                    Imgproc.line(img, prev.first, new Point(prev.first.x + dist, prev.first.y), new Scalar(0, 0, 255, 255), 1);
                }
                else if (dist > Config.fSize){
                    Imgproc.line(img, prev.first, new Point(prev.first.x + dist, prev.first.y), new Scalar(255, 0, 0, 255), 1);
                }
                else{
                    Imgproc.line(img, prev.first, new Point(prev.first.x + dist, prev.first.y), new Scalar(0, 255, 0, 255), 1);
                }
            }
            prev = curr;
        }
    }

    protected ArrayList<Rect> processFrame(Mat img, int kernel, Scalar hsvLower, Scalar hsvUpper){
        ArrayList<MatOfPoint> contour = new ArrayList<>();
        Mat h = new Mat();

        ArrayList<Rect> ret = new ArrayList<>();

        Imgproc.cvtColor(img, hsv, Imgproc.COLOR_RGB2HSV);
        Imgproc.medianBlur(hsv, hsvBlur, kernel);

        Core.inRange(hsvBlur, hsvLower, hsvUpper, bin);

//        Imgproc.morphologyEx(hsvBlur, hsvBlur, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, new Size(3, 3)));
        Imgproc.erode(bin, bin, getStructuringElement(MORPH_ELLIPSE, new Size(3, 3)));
        Imgproc.findContours(bin, contour, h, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for(MatOfPoint m : contour){
            ret.add(Imgproc.boundingRect(m));
        }

        return ret;
    }

    protected Point rescale(Point p){
        return new Point(p.x / (Config.scale / 100.0), p.y / (Config.scale / 100.0));
    }
}
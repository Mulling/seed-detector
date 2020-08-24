# Seed Detector

Software for detecting/measuring and statistics generation for seed distribution. 

For reference of how things work see the python version since its better documented.

### [Python](python/ "Python Prototype")

To install the python dependencies use:

```console
$ pip install -r requirements.txt
```
Usage:
```console
$ seedt [-f | --file=] [-k | --kernel=] [-s | --scale=]
```

Running 'test.sh' will start the tracker in a pre-configured state,
using one of the provided [videos](videos/ "Seed Videos").

```console
$ ./test.sh
```

There is also a script 'video_maker.py' to generate videos of
predetermined seed distributions, this is useful for testing the
tracker without the variables added by the extraction of the seed from
the image.

Flags:

* [`-f` | `--file=`] - The file name to be used as video source. Use `-d` to use a hardware device.
* [`-k` | `--kernel=`] - The size of the kernel to be used to blur the
  image, this number must be odd. Using a high kernel size will result
  in more blur and less noise on the image, but it Will result in a
  loss of sharpness.
* [`-s` | `--scale=`] - How much to scale the image down, 100 will
  result in 1:1 scale, 50 will result in 1:2 scale, and so on.
* [`-p` | `--pixel=`] - How many pixels are equal to 1cm. Defaults to 10.
* [`-F` | `--Fault=`] - The maximum size of a fault. Defaults to 5.
* [`-D` | `--Double=`] - The minimum size of a double. Defaults to 15.
* [`-c`] - Will start the program in calibration mode. It may not work
  very well if the size of the video is above HD, since the GUI options
  for opencv are very limited.
* [`-h`] - Use a hardware device indicated by `--file=`.
* [`-d` | `--dir=`] - The direction the videos is playing, or, the
  direction the camera is moving. Can be: up, down, left or right.

| key-bind | function                                                                 |
|:--------:|:-------------------------------------------------------------------------|
| q        | Exits the program.                                                       |
| p        | Pauses the video.                                                        |
| r        | Displays the results, this will happen automatically if the videos ends. |
| s        | Starts the program(if in calibration mode).                              |

**NOTE:** The higher the FPS of the video fed the to program the
better the results, also, if your camera has white-balance and
auto-focus, turn those off. The tracker right now is a little
sensitive, and requires a manual calibration to obtain good
results.

### [Android](android/ "Android Version")
![java](images/java.gif "java")

To build import the project in android-studio and go from there.

TODO: interface description

# About
Tracks an Aruco marker and logs the distance to a line. The line is defined by the user, by clicking on two points, which lies on the line, on the first frame of a video.
The log file is a csv file which is saved, as _analysis.csv_, in a folder with the same name as the video (without the file hint ending (e.g. ".mp4")). If the `--overlay` flag is used, the program will also produce a file named _overlay.mp4_, where the line and center of the Aruco marker is drawn ontop of the input video.
# Compile
**Only once** 

`cmake .`

**Every time afterwards** 

`make`

# Usage
See `./main --help`

# Libraries

## OpenCV
The OpenCV, OpenCV video and OpenCV contribution library is used.
All three can be installed by using apt on Ubuntu
`sudo apt install libopencv libopencv-contrib-dev libopencv-video-dev`

## Argparse
https://github.com/p-ranav/argparse

https://raw.githubusercontent.com/p-ranav/argparse/master/include/argparse/argparse.hpp
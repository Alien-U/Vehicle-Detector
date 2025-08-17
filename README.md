# Vehicle-Detector
# Vehicle Counter

This project uses Python and OpenCV to count vehicles passing through designated lanes in a video. It leverages background subtraction and contour detection to identify and track objects, classifying them into different lanes.

## Features

- **Vehicle Detection:** Utilizes the MOG background subtractor to isolate moving vehicles from the static background.
- **Lane-Based Counting:** Counts vehicles in three separate, predefined lanes.
- **Real-Time Visualization:** Displays the video with bounding boxes around detected vehicles and real-time counter updates for each lane.

## Prerequisites

Before running the script, ensure you have the following libraries installed:

- **OpenCV:** A library for computer vision tasks.
- **NumPy:** A library for numerical operations.

You can install these using pip:

```bash
pip install opencv-python
pip install numpy

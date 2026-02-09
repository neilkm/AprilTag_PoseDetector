# AprilTag_PoseDetector
Coding demo for OpenCV proof of concept for 2026 Robotics Eng. Capstone project.

Goal is to recognize the distance, rotation, and angle of a known size AprilTag placed
in front of the camera. 

App should show:
- a video stream with a box overlay around the AprilTag
- a live updating display of:
	- distance to the center of the AprilTag
	- rotation angle in X Y Z

# Getting Started
this project was built to run on M2 MacBookAir

run the following command to install dependencies:
$ pip install -r requirements.txt

calibrate camera (optional):
$ python3 src/camera_calibrate.py

run detect pose:
$ python3 src/detect_pose.py


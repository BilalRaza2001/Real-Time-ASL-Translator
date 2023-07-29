# Real-Time-ASL-Translator
An AI powered real-time American sign language translator for letters, Converting handsigns shown to letters which I created for my end of year university project.

The AI classifier is used in this program to power a quiz game with multiple gamemodes the goal of which was to create a fun way for users to learn ASL allowing people to communicate with people they otherwise couldn't

This program uses your webcam to detect handsigns and tries to classify them using openCV2, Mediapipe and Tensorflow 

# Requirements
OpenCV2
Numpy
Mediapipe
Tensorflow
## To train
Sklearn
Pandas
Seaborn

# Acknowledgements
This [repository](https://link-url-here.org) was used to collect the data which I heavily modified to work for my use

#Known Problems
The letters J and Z aren't included as these required motion in order to execute and wouldn't work with my approach, the letter V is also not included as this was too similar to other letters and was not functioning properly

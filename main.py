#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
from cv2 import putText
import numpy as np
import mediapipe as mp


from utils import CvFpsCalc
from utils import myTimer
from model import KeyPointClassifier



import random


#Global variables
letter = None
score = 0
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    """ Main function this will run everything"""
    global score
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    mp_drawing = mp.solutions.drawing_utils
    keypoint_classifier = KeyPointClassifier()


    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
   

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)





    #  ########################################################################

    #Initalizing variables
    mode = 0
    gameT = None
    t = None
    holdt = None
    pick_random_letter()
    simone = None
    while True:
        # if mode is normal
        if mode == 0:
            #if any timers are currently running they will be cancelled and the variable will be returned to none
            if gameT != None:
                gameT.cancel()
                gameT = None
            if t != None:
                t.cancel()
                t = None
            if holdt != None:
                holdt.cancel()
                holdt = None
        # If in a game gamemode 
        elif mode == 1 or mode == 2:
            #timer setup
            if gameT == None:
                gameT = myTimer.Timer(90,timerOver)
                gameT.start()
            #if timer got cancelled or ended
            if not gameT.is_alive():
                #return to normal mode
                mode = 0
            #if guess window has expired or cancelled
            if t == None or t.remaining() < 0:
                #restart it 
                t = start_timer()
                if mode == 2:
                    if simone != None:
                        # pick a new guesser
                        simone = getSimone()
        if mode == 2:
            if simone == None:
                # pick starting guesser

                simone = getSimone()
        fps = cvFpsCalc.get()
    
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                

              

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                #if the correct hand sign is being shown
                if check_hand_sign(keypoint_classifier_labels[hand_sign_id]):
                    #if not in normal mode
                    if mode == 1 or mode == 2:

                        if holdt == None:
                            #start the hold timer to make sure user holds sign for long enough
                            holdt = myTimer.Timer(5,timerOver)
                            holdt.start()
                     
                else:
                    # if another sign is shown during the hold window
                    if holdt != None:
                        #cancel the timer
                        holdt.cancel()
                        holdt = None

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(results,debug_image,mp_hands,mp_drawing)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )
        else:
            if holdt != None:
                #cancel the timer
                holdt.cancel()
                holdt = None

        #if hold timer has started
        if holdt != None:
            #and been run for longer than 5 sec
            if holdt.elapsed() >= 5:
                #if quiz mode
                if mode == 1:
                    score += 1
                #if Simone Says mode
                elif mode == 2:
                    #if asker was Simone remove a point if it was Pumba add a point
                    if simone == 1:
                        score -= 1 
                    else:
                        score += 1
                    #pick a new asker
                    simone = getSimone()
                #pick a new letter and cancel timer
                pick_random_letter()
                t.cancel()
        if t != None:
            #if user ignored what Simone asked for
            if t.elapsed() >= 15 and simone == 1:
                #add 2 points and extend the game timer by 20 sec
                score += 2
                if gameT != None:
                    timeRemaining  = gameT.remaining()
                    gameT.cancel()
                    gameT = myTimer.Timer(timeRemaining+20,timerOver)
                    gameT.start()
                #cancel round timer get new guesser and new letter
                t.cancel()
                simone = getSimone()
                pick_random_letter()
        if t != None:
            #if timer ended restart it pick a new letter and get a new guesser
            if not t.is_alive():
                t = start_timer()
                simone = getSimone()
        debug_image = draw_info(debug_image, fps, mode,t,gameT,simone)
        if score > 0:
            cv.putText(debug_image, "SCORE: "+str(score), (10, 150),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                   cv.LINE_AA) 
        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def start_timer():
    """ used to setup the game timer also calls pick_random_letter at the end of the timer"""
    t = myTimer.Timer(20, pick_random_letter)
    t.start()
    return t



def pick_random_letter():
    """ will pick a random letter from the labels in the labels for the dataset"""
    global letter
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]
    letter =  random.choice(keypoint_classifier_labels)

def check_hand_sign(shownHandSign):
    """ checks to see if the hand sign given matches the one asked for"""
    global letter
    if shownHandSign == letter:
        return True
    else:
        return False
def timerOver():
    """ this function is needed as I couldn't get timers to work without a function call so I made an empty function"""
    pass


def getSimone():
    """ picks a new guesser"""
    simoneList = [0,1]
    return random.choice(simoneList)

    
def select_mode(key, mode):
    "mode selector"
    global score
    if key == 110:  # n
        mode = 0
    if key == 113:  # q
        mode = 1
        score = 0
    if key == 115:  # s
        mode = 2
        score = 0
    return mode


def calc_bounding_rect(image, landmarks):
    "calculates dimensions for the bounding box I didn't write this came with original tool file"
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    "This will normalize all the landmark values I didn't write this came with original tool file"
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    """ this function will draw the bounding box I didn't write this came with original tool file"""
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,):
    "This will draw the prediction onto the bounding box I didn't write this came with original tool file"
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    return image


def draw_landmarks(landmarks,img,mp_hands,mp_drawing):
    if landmarks.multi_hand_landmarks:
        for index,hand in enumerate(landmarks.multi_hand_landmarks):
            mp_drawing.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,255),thickness=2,circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(255,44,255),thickness=2,circle_radius=4)
                                    )
        return img
    else:
        return img


def draw_info(image, fps, mode,t,gameT,simone):
    "This will draw all the text UI"
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)


    if mode == 0:
        cv.putText(image, "CONTROLS N: Normal, Q: Quiz, S: Simone says",(10,90),cv.FONT_HERSHEY_SIMPLEX,0.6,(255, 255, 255), 1,
                   cv.LINE_AA)
    if mode == 1:
        cv.putText(image, "MODE:" + "Quiz", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        cv.putText(image, "Show " + letter, (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                   cv.LINE_AA)     
    if mode == 2:
        simoneList = ["Pumba says ","Simone says "]
        cv.putText(image, "MODE:" + "Simone Says", (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                   cv.LINE_AA)
        cv.putText(image, "Gain a point for listening to Pumba and ignoring Simone", (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                   cv.LINE_AA)
        cv.putText(image, "Lose a point if you listen to Simone", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                   cv.LINE_AA)    
        if simone != None:
            cv.putText(image, simoneList[simone] + letter, (10, 130),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                    cv.LINE_AA) 
    if 1 <= mode <= 2:
        
        
        if t != None:
               cv.putText(image, "Guess Window: " + str(t.remaining()), (10, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                   cv.LINE_AA)
        
        if gameT != None:
            cv.putText(image, "Time Left: " + str(gameT.remaining()), (10, 170),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                    cv.LINE_AA) 
    return image


if __name__ == '__main__':
    main()

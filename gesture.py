import cv2
import numpy as np
import math
import time
from os import path
import pygame
from pygame.locals import *
import cPickle as pickle
from classifiers import MultiLayerPerceptron

def _on_exit(samples, labels,data_file='datasets/faces_training.pkl'):
        # if we have collected some samples, dump them to file
        print len(samples)
        if len(samples) > 0:
            # make sure we don't overwrite an existing file
            if path.isfile(data_file):
                # file already exists, construct new load_from_file
                load_from_file, fileext = path.splitext(data_file)
                offset = 0
                while True:
                    file = load_from_file + "-" + str(offset) + fileext
                    if path.isfile(file):
                        offset += 1
                    else:
                        break
                data_file = file
            print data_file 

            # dump samples and labels to file
            f = open(data_file, 'wb')
            pickle.dump(samples, f)
            pickle.dump(labels, f)
            f.close()

            print "Saved", len(samples), "samples to", data_file



#############################################################################
import cv2
import numpy as np
import copy
import math


# Environment:
# OS    : Mac OS EL Capitan
# python: 3.5
# opencv: 2.4.13

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works

def printThreshold(thr):
    print("! Changed threshold to "+str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow("hhg",fgmask)
    #kernel = np.ones((3, 3), np.uint8)
    #fgmask = cv2.erode(fgmask, kernel, iterations=2)
    cv2.imshow("yo",fgmask)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res






camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)





#pygame.init()
samples=[]
labels=[]
while(camera.isOpened()):

    '''ret, img = cap.read()
        
    cv2.rectangle(img,(400,400),(100,100),(0,255,0),0)
    crop_img = img[100:400, 100:400]
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresh1 = cv2.threshold(blurred, 111, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)
    cv2.imshow('Gesture', img)''' 
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    #cv2.imshow('yo',frame)
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]*1.3), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0]*0.7)), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]*0.7),
                    int(cap_region_x_begin * frame.shape[1]*1.3):frame.shape[1]]  # clip the ROI
       


        # convert the image into binary image
        thresh1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('mask', thresh1)

    #gamedisplay=pygame.display.set_mode((800,600))
    #pygame.display.update()
    '''for event in pygame.event.get():
                        #print event
                        if event.type == pygame.KEYDOWN:
                            #print "vubkvbvkj"
                            key_input = pygame.key.get_pressed()

                            if key_input[pygame.K_UP]:
                                text="victory"
                                print text
                                samples.append(thresh1.flatten())
                                labels.append(text)

                            elif key_input[pygame.K_DOWN]:
                                text="yo"
                                print text
                                samples.append(thresh1.flatten())
                                labels.append(text)
                            
                            elif key_input[pygame.K_RIGHT]:
                                text="like"
                                print text
                                samples.append(thresh1.flatten())
                                labels.append(text)

                            elif key_input[pygame.K_LEFT]:
                                text="none"
                                print text
                                samples.append(thresh1.flatten())
                                labels.append(text)

                            elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                                print 'exit'
                                _on_exit(samples,labels)
                                camera.release()
                                break
                                    
                        elif event.type == pygame.KEYUP:
                            print "1"'''


     
    k = cv2.waitKey(10)
    #print (k)
    
    if k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.BackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print '!!!Background Captured!!!'
    elif k == ord('r'):  # press 'r' to reset the background
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print '!!!Reset BackGround!!!'
    elif k == ord('n'):
        triggerSwitch = True
        print '!!!Trigger On!!!'


    elif k == ord('a'):
            text="Stone"
            print text
            samples.append(thresh1.flatten())
            labels.append(text)
    elif k == ord('s'):
            text="Paper"
            print text
            samples.append(thresh1.flatten())
            labels.append(text)
    elif k == ord('d'):
            text="Scissors"
            print text
            samples.append(thresh1.flatten())
            labels.append(text)
    elif k == ord('f'):
            text="Spock"
            print text
            samples.append(thresh1.flatten())
            labels.append(text)
    elif k == ord('g'):
            text="Lizard"
            print text
            samples.append(thresh1.flatten())
            labels.append(text)
    elif k == ord('h'):
            text="None"
            print text
            samples.append(thresh1.flatten())
            labels.append(text)
    elif k==ord('i'):
             print 'exit'
             _on_exit(samples,labels)
             camera.release()
             break
        

        
    elif k == 27:
        exit(0)


    

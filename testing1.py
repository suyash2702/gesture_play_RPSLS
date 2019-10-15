#this file is for gesture detection
import cv2
import numpy as np
import math
import time
from os import path
import pygame
from pygame.locals import *
import cPickle as pickle
from classifiers import MultiLayerPerceptron
from datasets import homebrew
import time
import urllib
import socket
from socket import *

'''new=socket()
new.connect(('192.168.43.34',7000))'''
#new.listen(0)
#a,addr=new.accept()

load_preprocessed_data='datasets/faces_preprocessed.pkl'
load_mlp='params/mlp.xml'

light1=0
light2=0
if path.isfile(load_preprocessed_data):
            (_, y_train), (_, y_test), V, m = homebrew.load_from_file(
                load_preprocessed_data)
            pca_V = V
            pca_m = m
            all_labels = np.unique(np.hstack((y_train, y_test)))

            # load pre-trained multi-layer perceptron
            if path.isfile(load_mlp):
                layer_sizes = np.array([pca_V.shape[1],
                                        len(all_labels)])
                MLP = MultiLayerPerceptron(layer_sizes, all_labels)
                MLP.load(load_mlp)



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
    #fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res






camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


############################################################




'''send_inst = True
stream=urllib.urlopen('http://192.168.1.1:8080/video')
stream_bytes = ' '
frame = 1
while self.send_inst:
    #stream_bytes += self.connection.read(1024)
    stream_bytes += self.stream.read(1024)
    first = stream_bytes.find('\xff\xd8')
    last = stream_bytes.find('\xff\xd9')
    if first != -1 and last != -1:
        jpg = stream_bytes[first:last + 2]
        stream_bytes = stream_bytes[last + 2:]
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD0_IMAGE_GRAYSCALE)'''



count=0
count1=0
second=""
first=""
output1=""
output2=""
player_one_win=0
player_two_win=0
tick=0
p1=[]
p2=[]
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
    #print (frame.shape[1]-(int(cap_region_x_begin * frame.shape[1]*1.3)))
    cv2.rectangle(frame, (0, 0),
                 (int(cap_region_x_begin * frame.shape[1]*0.7), int(cap_region_y_end * frame.shape[0]*0.7)), (255, 0, 0), 2)
    #print (int(cap_region_x_begin * frame.shape[1]*0.7))
    #print(frame[0:int(cap_region_y_end * frame.shape[0]*0.75),int(cap_region_x_begin * frame.shape[1]*0.75):int(cap_region_x_begin * frame.shape[1]*0.7)])
    frame[int(cap_region_y_end * frame.shape[0]*0.7):,:]=[0,0,0]
    frame[0:int(cap_region_y_end * frame.shape[0]*0.7),int(cap_region_x_begin * frame.shape[1]*0.7):int(cap_region_x_begin * frame.shape[1]*1.3)]=[0,0,0]
    #frr=frame[0:int(cap_region_y_end * frame.shape[0]*0.7),int(cap_region_x_begin * frame.shape[1]*0.75):int(cap_region_x_begin * frame.shape[1]*1.27)]

    #frame=frame[0:int(cap_region_y_end * frame.shape[0]*0.9),:]
    cv2.putText(frame, "Player 1", (10,int(cap_region_y_end * frame.shape[0]*0.8)),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, "Player 2", (int(cap_region_x_begin * frame.shape[1]*1.3)+10,int(cap_region_y_end * frame.shape[0]*0.8)),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    #  Main operation

    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img1 = img[0:int(cap_region_y_end * frame.shape[0]*0.7),
                    int(cap_region_x_begin * frame.shape[1]*1.3):frame.shape[1]]# clip the ROI
        img2=img[0:int(cap_region_y_end * frame.shape[0]*0.7),0:int(cap_region_x_begin * frame.shape[1]*0.7)]


        # convert the image into binary image
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        cv2.imshow('mask', gray)
        cv2.imshow('mask1', gray1)
        
        
	#print gray.shape
	#print gray1.shape
        

        X, _, _ = homebrew.extract_features([gray.flatten()],pca_V, pca_m)
        X1, _, _ = homebrew.extract_features([gray1.flatten()],pca_V, pca_m)
        
        
        label = MLP.predict(np.array(X))[0]
        label2 = MLP.predict(np.array(X1))[0]
        #print(label)
        #print str(label),first
        if tick%10==0:
            count=1
        else:
            p1.append(label)
	    count=0
        if count==1:
	    #print(str(label))
	    label=max(set(p1), key=p1.count)
	    print(label)
            if str(label)=="Stone":
                output1=str(label)
            elif str(label)=="Paper":
                output1=str(label)
            elif str(label)=="Scissors":
                output1=str(label)
            elif str(label)=="Spock":
                output1=str(label)
            elif str(label)=="Lizard":
                output1=str(label)
	    p1=[]

            #cv2.putText(img,"Command Performed!!", (150,450),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            pass
        first=str(label)
        if tick%10==0:
            count1=1
        else:
            count1=0
	    p2.append(label2)
        if count1==1:
	    #print(str(label2))
            label2=max(set(p2), key=p2.count)
	    print(label2)
            if str(label2)=="Stone":
                output2=str(label2)
            elif str(label2)=="Paper":
                output2=str(label2)
            elif str(label2)=="Scissors":
                output2=str(label2)
            elif str(label2)=="Spock":
                output2=str(label2)
            elif str(label2)=="Lizard":
                output2=str(label2)
	    p2=[]

            #cv2.putText(img,"Command Performed!!", (150,450),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            pass
        second=str(label2)

        if output1 and output2:
	    print(" ")
	    print(output1,output2)
            player_one_win=0
            player_two_win=0
            if output1=="Stone":
                if output2=="Paper":
                    player_one_win=0
                    player_two_win=1
                elif output2=="Scissors":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Spock":
                    player_one_win=0
                    player_two_win=1
                elif output2=="Lizard":
                    player_one_win=1
                    player_two_win=0
            elif output1=="Paper":
                if output2=="Stone":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Scissors":
                    player_one_win=0
                    player_two_win=1
                elif output2=="Spock":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Lizard":
                    player_one_win=0
                    player_two_win=1
            elif output1=="Scissors":
                if output2=="Paper":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Stone":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Spock":
                    player_one_win=0
                    player_two_win=1
                elif output2=="Lizard":
                    player_one_win=1
                    player_two_win=0
            elif output1=="Spock":
                if output2=="Paper":
                    player_one_win=0
                    player_two_win=1
                elif output2=="Scissors":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Stone":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Lizard":
                    player_one_win=0
                    player_two_win=1
            elif output1=="Lizard":
                if output2=="Paper":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Scissors":
                    player_one_win=0
                    player_two_win=1
                elif output2=="Spock":
                    player_one_win=1
                    player_two_win=0
                elif output2=="Stone":
                    player_one_win=0
                    player_two_win=1
	    output1=""
            output2=""
        if player_one_win==1:
            cv2.putText(frame,"Player2 Won", (200,400),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
 	    #print "Player 2 wins"       
	elif player_two_win==1:
            cv2.putText(frame, "Player1 Won", (200,400),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
	    #print "Player 1 wins"

            
            


        

        cv2.putText(frame, str(label), (500,100),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
	cv2.putText(frame, str(label2), (50,100),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    #cv2.imshow('Gesture', img)
    cv2.imshow('original', frame)
    #time.sleep(0.5)
    tick+=1
    k=cv2.waitKey(50)
    if k==27:
	    break
    elif k == ord('b'):  # press 'b' to capture the background
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



cap.release()
cv2.destroyAllWindows()










'''while(cap.isOpened()):

    ret, img = cap.read()
        
    cv2.rectangle(img,(300,300),(100,100),(0,255,0),2)
    crop_img = img[100:300, 100:300]
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurred = cv2.GaussianBlur(grey, value, 0)
    #cv2.imshow('blur',blurred)
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Thresholded', thresh1)
    cv2.imshow('Gesture', img)'''

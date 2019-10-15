#cheers to hacktoberfest
import cv2
import numpy as np
import math
import time
from os import path
import pygame
from pygame.locals import *
import cPickle as pickle
from classifiers1 import MultiLayerPerceptron
from datasets1 import homebrew
import time
import urllib
import socket
from socket import *
from Tkinter import *
import os
import subprocess
#import pyttsx
chamru = 0
#speech_engine = pyttsx.init() # see http://pyttsx.readthedocs.org/en/latest/engine.html#pyttsx.init
#speech_engine.setProperty('rate', 150)

#def speak(text):
#	speech_engine.say(text)
#	speech_engine.runAndWait()
'''file = 'music_01.mp3'
root = Tk()
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(file)
pygame.mixer.music.play()
root.mainloop()'''

'''new=socket()
new.connect(('192.168.43.34',7000))'''
#new.listen(0)
#a,addr=new.accept()

load_preprocessed_data='datasets1/faces_preprocessed.pkl'
load_mlp='params1/mlp.xml'

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

pygame.init()
white = (255,255,255)
black = (0,0,0)
light_red = (255,0,0)
red = (255,50,0)
light_green = (0,255,0)
green = (0,155,0)
yellow = (200,200,0)
light_yellow = (255,255,0)

display_width  = 400
display_height = 600

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('RPSLS')

clock = pygame.time.Clock()
smallfont = pygame.font.Font(None,25)
medfont = pygame.font.Font(None,50)
largefont = pygame.font.Font(None,80)
superlargefont = pygame.font.Font(None,120)


class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

BackGround = Background('sprites/intro.png', [0,0])
#gameDisplay.fill(white)
gameDisplay.fill([255, 255, 255])
gameDisplay.blit(BackGround.image, BackGround.rect)
pygame.display.update()

def text_objects(text,color,size):
	if size == "small":
		textSurface = smallfont.render(text,True,color)
	elif size == "medium":
		textSurface = medfont.render(text,True,color)
	elif size == "large":
		textSurface = largefont.render(text,True,color)
	elif size == "super_large":
		textSurface = superlargefont.render(text,True,color)
	return textSurface, textSurface.get_rect()

def button(x,y,width,height,action):
	cur = pygame.mouse.get_pos()

	click = pygame.mouse.get_pressed()
	if x + width > cur[0] > x and y + height > cur[1] > y:
		#pygame.draw.rect(gameDisplay,active_color,(x,y,width,height))
		if click[0] == 1 and action != None:
			if action == "instructions":
				rules()
			if action == "single_play":
				pass
			if action == "multi_play":
				gameLoop("multi_play")
			if action == "back_to_main":
				pass
	#text_to_button(text,black,x,y,width,height)

def active_button(text,x,y,width,height,action):
	cur = pygame.mouse.get_pos()
	click = pygame.mouse.get_pressed()
	if x + width > cur[0] > x and y + height > cur[1] > y:
		pygame.draw.rect(gameDisplay,red,(x,y,width,height))
		if click[0] == 1 and action != None:
			if action == "back_to_menu":
				return(action)
			if action == "more":
				knockout()
	else:
		pygame.draw.rect(gameDisplay,light_red,(x,y,width,height))
	text_to_button(text,black,x,y,width,height)

def text_to_button(msg,color,buttonx,buttony,buttonwidth,buttonheight,size = "small"):
	textSurface,textRect = text_objects(msg,color,size)
	textRect.center = ((buttonx+(buttonwidth/2)),buttony+(buttonheight/2))	
	gameDisplay.blit(textSurface,textRect)

def message_to_screen(msg,color,y_displace=0,size = "small"):
	textSurface,textRect = text_objects(msg,color,size)
	textRect.center = (display_width/2),(display_height/2)+y_displace
	gameDisplay.blit(textSurface,textRect)

def credits():

	cred = True
	while cred:
		for event in pygame.event.get():
			#print(event)
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
		BackGround = Background('sprites/karlo.jpg', [0,0])
		gameDisplay.fill([255, 255, 0])
		gameDisplay.blit(BackGround.image, BackGround.rect)		
		#gameDisplay.fill(white)
		'''message_to_screen("KNOCKOUT",light_red,-200,"large")
		#message_to_screen("Compete in a knockout tournament of",black,-120,"medium")
		#message_to_screen("RPSLS with your friends.",black,-80,"medium")
		message_to_screen("==> A total of 4 players participate",green,-90,"small")
		message_to_screen("in this tournament.",green,-60,"small")
		message_to_screen("==> Only 1 player advances from each round.",green,30,"small")
		message_to_screen("==> The player to score 5 points first wins",green,0,"small")
		button(550,0,50,50,action = "instructions")
		'''
		out = active_button("Back",350,30,50,50,action = "back_to_menu")
		if out == "back_to_menu":
			break
		pygame.display.update()

def gameLoop():
	loop = 3
	while loop > 0:
		for event in pygame.event.get():
			#print 'hfrsts'
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
		gameDisplay.fill(light_yellow)		
		message_to_screen(str(loop),green,0,"large")
		pygame.display.update()
		time.sleep(1)
		loop -= 1
	gameDisplay.fill([255, 255, 255])
	message_to_screen('GO',green,0,"large")
	pygame.display.update()
	time.sleep(1)
	os.system("testing1.py")

def rules():

	gcont = True
	while gcont:
		for event in pygame.event.get():
			#print(event)
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
		BackGround = Background('sprites/images1.jpeg', [0,0])
		#gameDisplay.fill(white)
		gameDisplay.fill([255, 255, 255])
		gameDisplay.blit(BackGround.image, BackGround.rect)
		'''message_to_screen("Instructions",light_red,-200,"large")
		message_to_screen("Use Hand Gestures to form figures of:",black,-80,"medium")
		message_to_screen("==> ROCK",green,-30,"small")
		message_to_screen("==> PAPER",green,0,"small")
		message_to_screen("==> SCISSORS",green,30,"small")
		message_to_screen("==> LIZARD",green,60,"small")
		message_to_screen("==> SPOCK",green,90,"small")
		'''
		out = active_button("Back",350,30,50,50,action = "back_to_menu")
		active_button("More",0,30,50,50,action = "more")
		if out == "back_to_menu":
			break

		pygame.display.update()
	
'''def game_intro():
	intro = True
	while intro:
		for event in pygame.event.get():
			#print(event)
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
		BackGround = Background('sprites/intro.png', [0,0])
		#gameDisplay.fill(white)
		gameDisplay.fill([255, 255, 255])
		gameDisplay.blit(BackGround.image, BackGround.rect)
		message_to_screen("Welcome to RPSLS",green,-100,"large")
		message_to_screen("Play a game of Rock Paper Scissor Lizard Spock",black,-30)
		message_to_screen("BAZINGA",red,30,"large")
		
		button(50,450,100,50,action = "single_play")
		button(200,450,150,50,action = "multi_play")
		#button("quit",550,500,100,50,red,light_red,action = "quit")
		button(350,0,50,50,action = "instructions")
		pygame.display.update()
'''

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
    #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("hhg",fgmask)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
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



counting1 = 0
counting2 = 0 
quit_game=0
first=""
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
    gameDisplay = pygame.display.set_mode((display_width,display_height))
    pygame.display.set_caption('RPSLS')
    BackGround = Background('sprites/intro.png', [0,0])
    gameDisplay.fill([255, 255, 255])
    gameDisplay.blit(BackGround.image, BackGround.rect)
    pygame.display.update()


    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    #cv2.imshow('yo',frame)
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]*1.3), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0]*0.7)), (255, 0, 0), 2)
    

    #  Main operation
    if isBgCaptured == 1:  # this part wont run until background captured
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]*0.7),
                    int(cap_region_x_begin * frame.shape[1]*1.3):frame.shape[1]]  # clip the ROI
       


        # convert the image into binary image
        thresh1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('mask', thresh1)

        X, _, _ = homebrew.extract_features([thresh1.flatten()],pca_V, pca_m)
        
        label = MLP.predict(np.array(X))[0]
        
        #print str(label),first
        if str(label)==first:
            count+=1
        else:
            count=0
        '''if count==12:
            if str(label)=="victory":
                if light1==0:
                    #new.send("  light1_on")
                    light1=1
                elif light1==1:
                    #new.send("  light1_off")
                    light1=0
            if str(label)=="yo":
                if light1==0:
                    #new.send("  light2_on")
                    light1=1
                elif light1==1:
                    #new.send("  light2_off")
                    light1=0
            if str(label)=="like":
                    #new.send("  light1_off")
                    light1=0
                    #new.send("  light2_off")
                    light1=0
            count=0
            if light1==1:
                print "1 is on"
            elif light2==1:
                print "2 is on"
            elif light1==0 and light2==0:
                print "no light is on "
            cv2.putText(img,"Command Performed!!", (150,450),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        else:
            print '1'''
  

        cv2.putText(frame, str(label), (100,100),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    	print str(label)
	if count==10:
	    	if str(label) == "instructions":
	    		rules()
	    	if str(label) == "credits":
	    		credits()
		if str(label) == "play":
			break
		if str(label) == "quit":
			pygame.quit()
			quit_game=1
			break	
	first=str(label)
    cv2.imshow('original', frame)
    #cv2.imshow('Gesture', img)
    #time.sleep(0.5)
    
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
    

camera.release()
cv2.destroyAllWindows()





####################################################3

if quit_game==0:
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
	speak("Press B to start")
	#speech_engine.runAndWait()
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
		    print("output")
		    chamru = 1
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
			else:
			    pass
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
			else:
			    pass
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
			else:
			    pass
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
			else:
			    pass
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
		    	else:
			    pass
		    else:
			chamru = 0
		        pass
		    output1=""
		    output2=""
				
		if player_one_win==1:
		    cv2.putText(frame,"Player2 Won", (200,400),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
	 	    if chamru == 1:
		    	speak("Player two wins")
			#speech_engine.runAndWait()		        
			chamru = 0
			counting1 += 1		        			
			#print "Player 2 wins"       
		elif player_two_win==1:
		    cv2.putText(frame, "Player1 Won", (200,400),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
		    if chamru == 1:		    
			speak("Player one wins")			
			#speech_engine.runAndWait()			
			chamru = 0	
			counting2 += 1	    
			#print "Player 1 wins"

		    
		    

		cv2.putText(frame,"SCORE"+str(counting2),(20,350),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
		cv2.putText(frame,"SCORE"+str(counting1),(400,350),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
		print "count1::",counting1,"    count2::",counting2
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










	

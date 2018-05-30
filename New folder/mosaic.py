'''https://www.kaggle.com/abcsds/pokemon/kernels/forknb/455476'''

import numpy 
import cv2
import random

print random.randrange(-10,10)

camera = cv2.VideoCapture(0)
flag=1
array=[]
while True:
        
        grabbed, frame = camera.read()
        shape=frame.shape[:2]
        cv2.rectangle(frame,(300,300),(100,100),(0,255,0),2)
        
        centre=(shape[1]/2,shape[0]/2)
        yo = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #value = (35, 35)
        #blurred = cv2.GaussianBlur(grey, value, 0)
        
        #print centre
        
        if flag==1:
                for i in range(10):
                        final_centre=(centre[0]+random.randrange(-80,80),centre[1]+random.randrange(-150,150))
                        roi=yo[final_centre[1]:final_centre[1]+20,final_centre[0]:final_centre[0]+20]
                        average_color_per_row = numpy.average(roi, axis=0)
                        average_color = numpy.average(average_color_per_row, axis=0)
                        #print average_color

                        array.append(average_color)
                        cv2.rectangle(frame,(final_centre[0]+20,final_centre[1]+20),(final_centre[0],final_centre[1]),(0,255,0),2)
        else:
                print"yo"
        new_array=[]
        for average_color in array:
                average_color[0]=int(average_color[0])
                average_color[1]=int(average_color[1])
                average_color[2]=int(average_color[2])
                lower = numpy.array([average_color[0]-20, average_color[1]-20, average_color[2]-20], dtype = "uint8")
                upper = numpy.array([average_color[0]+20, 255, 255], dtype = "uint8")
                print lower,upper
                skinMask = cv2.inRange(yo, lower, upper)
                new_array.append(skinMask)
        skinMask=sum(new_array)
        cv2.imshow("yo",frame)
        cv2.imshow("yo1",skinMask)                            
        #print numpy.average(array,axis=0)
        #crop_img = blurred[100:300, 100:300]
        #cv2.imshow('blur',blurred)
        #_, thresh1 = cv2.threshold(frame, 140, 255,cv2.THRESH_BINARY)
        
        #skinMask = cv2.inRange(yo, lower, upper)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        #skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        #skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        #skin = cv2.bitwise_and(frame, frame, mask = skinMask)
        #cv2.imshow("yo1",thresh1)
        flag=0
        k = cv2.waitKey(10)
        if k == 27:
                exit(0)
cv2.destroyAllWindows()

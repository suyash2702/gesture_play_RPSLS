import cv2
import numpy as np

def run_grabcut(img_orig, rect_final):
    mask = np.zeros(img_orig.shape[:2],np.uint8)
    x,y,w,h = rect_final
    mask[y:y+h, x:x+w] = 1
    #print mask

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    cv2.grabCut(img_orig,mask,rect_final,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    #cv2.imshow('Output1', img_orig)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_orig = img_orig*mask2[:,:,np.newaxis]

    cv2.imshow('Output', img_orig)


camera = cv2.VideoCapture(0)
flag=0
while True:
        flag+=1   
        grabbed, frame = camera.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)
        shape=frame.shape[:2]
        if flag<=1111:
            cv2.rectangle(frame,(300,300),(100,100),(0,255,0),2)
        rect_final=(100,100,200,200)
        #run_grabcut(blurred, rect_final)
        
        cv2.imshow('Input', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

cv2.destroyAllWindows()


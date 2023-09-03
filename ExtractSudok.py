import cv2
import numpy as np
#from tensorflow.keras.models import load_model
import imutils
#from solver import *
import os

import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import sys


os.chdir('/home/micka/Desktop/Sudoku/sudoku/Images')
img = cv2.imread('sudoku.jpeg')

def find_board(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 100, 255, 0)
    #thresh = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.imshow("Contour", thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    keypoints= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    #cv2.imshow("Contour", newimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #contours=sorted(contours, key=cv2.contourArea, reverse = True)[:15]
    contours=sorted(contours, key=cv2.contourArea, reverse = True)[:15]
    location=None
    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 20, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location)
    return result, location

def get_perspective(img, location, height = 900, width = 900):
    print(location)
    """Takes an image and location of an interesting region.
    And return the only selected region with a perspective transformation"""
    tl, tr, bl, br = order_loc(location)
    pts1 = np.float32([tl, tr, bl, br])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def order_loc(temploc):

    sumA=[]
    coor1=[]
    coor2=[]
    for loc in temploc:
        sumA.append(loc[0][0]+loc[0][1])

    sumA=np.array(sumA)
    top_left=temploc[np.argmin(sumA)]
    bottom_right=temploc[np.argmax(sumA)]
    temploc=np.delete(temploc,np.argmax(sumA),0)
    sumA=np.delete(sumA,np.argmax(sumA))
    temploc=np.delete(temploc,np.argmin(sumA),0)


    for loc in temploc:
        coor1.append(loc[0][0])
    coor1=np.array(coor1)

    top_right=temploc[np.argmax(coor1)]
    temploc=np.delete(temploc,np.argmax(coor1),0)
    bottom_left=temploc[0]
    return top_left, top_right, bottom_left, bottom_right


def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells.
    each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            box = cv2.resize(box, (100, 100))/255.0
            #cv2.imshow("Splitted block", box)
            #cv2.waitKey(50)
            boxes.append(box)
    #cv2.imshow("Box", boxes[1])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()    
    return boxes


def post_treatment_boxes(boxes):
    cleanBoxes=[]
    for box in boxes:
        cropped_image = np.uint8(box[15:85, 15:85]*255)
        imgray_CI = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        #thresh=imgray_CI
        ret, thresh = cv2.threshold(imgray_CI, 140, 255, 0)
        threshComp=cv2.resize(thresh,(28,28), interpolation = cv2.INTER_AREA)
        threshComp=abs(threshComp/255-1)
        cleanBoxes.append(threshComp)
    cleanBoxes=np.asarray(cleanBoxes)
    
    return cleanBoxes





board, location = find_board(img)

boxes= split_boxes(board)

cleanBoxes=post_treatment_boxes(boxes)

#for testing:
cv2.imshow("Box", cleanBoxes[10])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Box", cleanBoxes[22])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("Box", cleanBoxes[33])
cv2.waitKey(0)
cv2.destroyAllWindows()
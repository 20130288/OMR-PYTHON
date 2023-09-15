import cv2
import numpy as np


def rectContours(contours):
    rectList = []
    for i in contours:
        area = cv2.contourArea(i)
        if (area > 50):
            parameter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * parameter, True)
            if (len(approx) == 4):
                rectList.append(i)

    rectList = sorted(rectList, key=cv2.contourArea, reverse=True)

    return rectList


def getConnerPoint(contour):
    parameter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * parameter, True)
    return approx


def reOrderPoint(point):
    point = point.reshape((4, 2))
    orderPoint = np.zeros((4, 1, 2), np.int)
    sumPoint = point.sum(1)
    orderPoint[0] = point[np.argmin(sumPoint)]
    orderPoint[3] = point[np.argmax(sumPoint)]
    diffPoint = np.diff(point)
    orderPoint[1] = point[np.argmin(diffPoint)]
    orderPoint[2] = point[np.argmax(diffPoint)]
    return orderPoint


def splitChoiceBoxes(image):
    rows = np.vsplit(image, 5)
    boxes = []
    for i in rows:
        cols = np.hsplit(i, 4)
        for box in cols:
            boxes.append(box)
    return boxes


def splitIdBoxes(image):
    cols = np.hsplit(image, 8)
    boxes = []
    for i in cols:
        rows = np.vsplit(i, 10)
        for box in rows:
            boxes.append(box)

    return boxes


def splitCodeBoxes(image):
    cols = np.hsplit(image, 3)
    boxes = []
    for i in cols:
        rows = np.vsplit(i, 10)
        for box in rows:
            boxes.append(box)

    return boxes

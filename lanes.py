import cv2                                               #Importing opencv into the code
import numpy as np                                       #importing numpy as np for numerical calculations\

def make_coordinates(lane_image, line_parameters):
    slope, intercept = line_parameters
    y1 = lane_image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int ((y1 - intercept)/slope)
    x2 = int ((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(lane_image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(lane_image, left_fit_average)
    right_line = make_coordinates(lane_image, right_fit_average)
    return np.array([left_line, right_line])

def canny_edge(lane_image):
    gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)#Using opencv in-built function cvt for color conversion from color to gray, this coonversion takes place because of the RGB is 3 channel and gray scale is 1 channel
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)     #Using opencv in-built function GaussianBlur we're smoothening the image: cv2.GaussianBlur(image, (size of the kernel, like (5, 5), Deviation))
    canny_edge = cv2.Canny(blur_image, 50, 150)              #Using canny edge detection we're detecting the edges between high threshold and low threshold pixels cv2.Canny(image, low_threshold, high_threshold) 1:3 for low_threshold to high_threshold
    return canny_edge

def display_lines(lane_image, lines):
    line_image = np.zeros_like(lane_image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(lane_image):
    height = lane_image.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(lane_image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(lane_image, mask)
    return masked_image


"""
# Finding Lanes in image, which is already loaded in the specific folder to code
image = cv2.imread('test_image.jpg')                     #Reading the image for the same folder where the code is located and assigning it to Variable image
lane_image = np.copy(image)                              #Using numpy as np function we're copying image from the variable image to lane_image
canny_image = canny_edge(lane_image)
region =  region_of_interest(canny_image)
lines = cv2.HoughLinesP(region, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=4)
averaged_lines = average_slope_intercept(lane_image, lines)
#line_image = display_lines(lane_image, lines)
line_image = display_lines(lane_image, averaged_lines)
integ_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#cv2.imshow('Original', lane_image)
#cv2.imshow('Gray Image', gray_image)
#cv2.imshow('Gaussianblur', blur_image)
#cv2.imshow('Region of Interest', region)
#cv2.imshow('Hough Transform', line_image)
cv2.imshow('Combo Image', integ_image)
#cv2.imshow('Combo Image', line_image)
cv2.waitKey(0)                                           #Untill any key pressed the image won't close

"""

#Finding lanes in Video, which is already loaded in the specific folder to code
capture = cv2.VideoCapture("test2.mp4")
while(capture.isOpened()):
    _, frame = capture.read()
    canny_image = canny_edge(frame)
    region =  region_of_interest(canny_image)
    lines = cv2.HoughLinesP(region, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=4)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    integ_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('Combo Image', integ_image)
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

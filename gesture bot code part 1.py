# organize imports
import cv2
import imutils
import numpy as np

# global variables
bg = None

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, bx) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
    
#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
#qcamera.release()
cv2.destroyAllWindows()





def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count


def segment(image, grayimage, threshold=75):
    # threshold the image to get the foreground which is the hand
    thresholded = cv2.threshold(grayimage, threshold, 255, cv2.THRESH_BINARY)[1]
    print("Original image shape - " + str(image.shape))
    print("Gray image shape - " + str(grayimage.shape))

    # show the thresholded image
    cv2.imshow("Thresholded", thresholded)

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # analyze the contours
        print("Number of Contours found = " + str(len(cnts))) 
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
        cv2.imshow('All Contours', image) 
        
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        cv2.drawContours(image, segmented, -1, (0, 255, 0), 3)
        cv2.imshow('Max Contour', image) 
        
        return (thresholded, segmented)


print("Type of Contour: " + str(type(segmented)))
print("Contour shape: " + str(segmented.shape))
print("First 5 points in contour: " + str(segmented[:5]))
# find the convex hull of the segmented hand region
chull = cv2.convexHull(segmented)

print("Type of Convex hull: " + str(type(chull)))
print("Length of Convex hull: " + str(len(chull)))
print("Shape of Convex hull: " + str(chull.shape))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.imshow("Convex Hull", image)
print(chull[:,:,1])
print(chull[:,:,1].argmin())
print(chull[chull[:,:,1].argmin()])
print(chull[chull[:,:,1].argmin()][0])
print(tuple(chull[chull[:,:,1].argmin()][0]))
# find the most extreme points in the convex hull
extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

print("Extreme Top : " + str(extreme_top))
print("Extreme Bottom : " + str(extreme_bottom))
print("Extreme Left : " + str(extreme_left))
print("Extreme Right : " + str(extreme_right))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.circle(image, extreme_top, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_bottom, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_left, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_right, radius=5, color=(0,0,255), thickness=5)
cv2.imshow("Extreme Points in Convex Hull", image)
# find the center of the palm
cX = int((extreme_left[0] + extreme_right[0]) / 2)
cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
print("Center point : " + str(tuple((cX,cY))))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.circle(image, (cX, cY), radius=5, color=(255,0,0), thickness=5)
cv2.circle(image, extreme_top, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_bottom, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_left, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_right, radius=5, color=(0,0,255), thickness=5)
cv2.imshow("Extreme Points in Convex Hull", image)
# find the maximum euclidean distance between the center of the palm
# and the most extreme points of the convex hull
distances = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
max_distance = distances[distances.argmax()]

# calculate the radius of the circle with 80% of the max euclidean distance obtained
radius = int(0.8 * max_distance)

# find the circumference of the circle
circumference = (2 * np.pi * radius)

print("Euclidean Distances : " + str(distances))
print("Max Euclidean Distance : " + str(max_distance))
print("Radius : " + str(radius))
print("Circumference : " + str(circumference))
# initialize circular_roi with same shape as thresholded image
circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
print("Circular ROI shape : " + str(circular_roi.shape))
cv2.imshow("Thresholded", thresholded)

# draw the circular ROI with radius and center point of convex hull calculated above
cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
cv2.imshow("Circular ROI Circle", circular_roi)

# take bit-wise AND between thresholded hand using the circular ROI as the mask
# which gives the cuts obtained using mask on the thresholded hand image
circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
cv2.imshow("Bitwise AND", circular_roi)
# compute the contours in the circular ROI
(_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of Contours found = " + str(len(cnts))) 
cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
cv2.imwrite("resources/count-contours.jpg", image)
cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
print("Count of fingers : " + str(len(cntsSorted[1:])))
count = 0

# loop through the contours found
for c in cnts:
    print(type(c))
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(image,'C' + str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 4)

    # increment the count of fingers only if -
    # 1. The contour region is not the wrist (bottom area)
    # 2. The number of points along the contour does not exceed
    #     25% of the circumference of the circular ROI
    if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
        count += 1

print("Count of fingers : " + str(len(cntsSorted[1:])))
cv2.imshow("Contours of fingers with box", image)
from scipy.spatial import distance as dist
from imutils import face_utils
from urllib.request import urlopen
import imutils
import numpy as np
import dlib
import cv2
import winsound

frequency = 2500 #frequency of the beeping sound
duration = 1000 #duration of the beeping sound

#function for identifying eye aspect ratio
def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

count = 0
earThresh = 0.2 #distance between vertical eye coordinate Threshold
earFrames = 4 #consecutive frames for eye closure

#Loading pre-trained 68 face landmark detector
shapePredictor = "shape_predictor_68_face_landmarks.dat"

#Using IP WebCam to use mobile camera 
url='http://192.168.43.129:8080/shot.jpg?rnd=632785' #change this with your IP WebCam link

#Loading Face detector & Landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

#geting the coordinates of left & right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    #Reading from the camera(mobile camera using IP WebCam)
    imgResp=urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)

    frame = imutils.resize(img, width=800)
    #Converting RGB image to gray scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #getting eye coordinates and eye aspects ratio
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye) 
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #drawing contours around the eyes
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)

        #comparing the average of eye aspect ratio of both the eyes with our intital threshold
        if ear < earThresh:
            count += 1
            print("Eye closed!")
            if count >= earFrames:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(frequency, duration)
        else:
            count = 0

    #showing live camera feed
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):#exiting the feed when q is pressed
        break


cv2.destroyAllWindows()


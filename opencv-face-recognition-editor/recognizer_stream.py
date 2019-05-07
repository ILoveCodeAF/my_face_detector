# USAGE
# python recognize.py --detector face_detection_model \
# 	--embedding-model openface_nn4.small2.v1.t7 \
# 	--recognizer output/recognizer.pickle \
# 	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import cv2
from Recognizer_image import RecognizerImage

cam = cv2.VideoCapture(0)
detector = RecognizerImage()
while True:
    ret, frame = cam.read()
    detector.setImage(frame)

    #isflip = True and flipCode = 1
    frame = detector.detect()

    cv2.imshow("image", frame)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()


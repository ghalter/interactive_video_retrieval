"""
Here we preprocess all data, store it in some database / hdf5 file, for later retrieval

"""

import cv2


cap = cv2.VideoCapture("my_file.mp4")

ret = True
while ret is True:
    ret, frame = cap.read()
    cv2.imshow("output", frame)
    cv2.waitKey(10)
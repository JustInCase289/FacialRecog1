# OpenCV program to detect face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2
import time

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalface_default.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# https://github.com/Itseez/opencv/blob/master
# /data/haarcascades/haarcascade_eye.xml
# Trained XML file for detecting eyes
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)

# Initialize variables for frame rate calculation
frame_count = 0
start_time = time.time()

# loop runs if capturing has been initialized.
while 1:

    # reads frames from a camera
    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detects the side profile of face
    side_faces = profile_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # To draw a rectangle in eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 200, 300), 2)

            # Apply Gaussian blur to the face and eyes ROIs
            roi_color_blurred = cv2.GaussianBlur(roi_color, (15, 15), 75)
            img[y:y + h, x:x + w] = roi_color_blurred

            # Apply Bilateral filter to the face and eyes ROIs
            # roi_color_blurred = cv2.bilateralFilter(roi_color, 5, 5, 5)  # Adjust parameters as needed
            # img[y:y + h, x:x + w] = roi_color_blurred

    # Display an image in a window
    cv2.imshow('img', img)

    # Increment the frame count
    frame_count += 1

    # Calculate and display frame rate every 30 frames (adjust as needed)
    if frame_count % 30 == 0:
        end_time = time.time()
        elapsed_time = end_time - start_time
        frame_rate = frame_count / elapsed_time
        print(f"Frame Rate: {frame_rate:.2f} fps")
        frame_count = 0
        start_time = time.time()

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import cv2

# Load the required trained XML classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

# Load an image from file
img = cv2.imread('C:\\School\\599\\StockImageGroup.jpg')

# Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
profile_faces = profile_cascade.detectMultiScale(gray, 1.3, 5)
# Detect faces with different minNeighbors
faces_3n = face_cascade.detectMultiScale(gray, 1.3, 3)
faces_5n = face_cascade.detectMultiScale(gray, 1.3, 5)
faces_7n = face_cascade.detectMultiScale(gray, 1.3, 7)


# Combine faces and profile faces detections
all_faces = list(faces) + list(profile_faces)

for (x, y, w, h) in all_faces:
    # Blur the face region
    face_region = img[y:y + h, x:x + w]
    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
    img[y:y + h, x:x + w] = blurred_face

    # Check the "confidence"
    if (x, y, w, h) in faces_7n:
        confidence_text = "90%"
    elif (x, y, w, h) in faces_5n:
        confidence_text = "70%"
    else:
        confidence_text = "50%"

    # Display confidence on the image
    cv2.putText(img, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # If you want to detect and blur eyes within frontal faces, do that within this loop
    if (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_region = roi_color[ey:ey + eh, ex:ex + ew]
            blurred_eye = cv2.GaussianBlur(eye_region, (49, 49), 30)
            roi_color[ey:ey + eh, ex:ex + ew] = blurred_eye

# Display an image in a window
cv2.imshow('img', img)

# Wait for Esc key to stop
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

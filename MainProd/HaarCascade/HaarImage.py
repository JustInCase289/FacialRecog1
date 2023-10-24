import time

import cv2

# Load the required trained XML classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load an image from file
img = cv2.imread('C:\\Users\\bisho\\PycharmProjects\\FacialRecog\\MainProd\\TestImages\\StockImageGroup4.jpg')
# Provide the path to your image


# Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces with different minNeighbors
faces_3n = face_cascade.detectMultiScale(gray, 1.3, 3)
faces_5n = face_cascade.detectMultiScale(gray, 1.3, 5)
faces_7n = face_cascade.detectMultiScale(gray, 1.3, 7)

# Detect left-looking profile faces
profiles_left_3n = profile_cascade.detectMultiScale(gray, 1.3, 3)
profiles_left_5n = profile_cascade.detectMultiScale(gray, 1.3, 5)
profiles_left_7n = profile_cascade.detectMultiScale(gray, 1.3, 7)

# Flip the image horizontally
flipped_img = cv2.flip(img, 1)
flipped_gray = cv2.cvtColor(flipped_img, cv2.COLOR_BGR2GRAY)

# Detect right-looking profile faces
profiles_right_3n = profile_cascade.detectMultiScale(flipped_gray, 1.3, 3)
profiles_right_5n = profile_cascade.detectMultiScale(flipped_gray, 1.3, 5)
profiles_right_7n = profile_cascade.detectMultiScale(flipped_gray, 1.3, 7)

# Flip the coordinates back
profiles_right_3n = [(img.shape[1] - x - w, y, w, h) for (x, y, w, h) in profiles_right_3n]
profiles_right_5n = [(img.shape[1] - x - w, y, w, h) for (x, y, w, h) in profiles_right_5n]
profiles_right_7n = [(img.shape[1] - x - w, y, w, h) for (x, y, w, h) in profiles_right_7n]

# Combine all detections
all_faces_3n = list(faces_3n) + list(profiles_left_3n) + list(profiles_right_3n)
all_faces_5n = list(faces_5n) + list(profiles_left_5n) + list(profiles_right_5n)
all_faces_7n = list(faces_7n) + list(profiles_left_7n) + list(profiles_right_7n)

# Combine all detections
all_faces = [(x, y, w, h, "90%") for x, y, w, h in all_faces_7n] + \
            [(x, y, w, h, "70%") for x, y, w, h in all_faces_5n] + \
            [(x, y, w, h, "50%") for x, y, w, h in all_faces_3n]

processed_faces = []


def is_face_processed(x, y, w, h):
    for px, py, pw, ph in processed_faces:
        if abs(px - x) < w / 2 and abs(py - y) < h / 2:
            return True
    return False


# Start the timer
start_time = time.time()


def process_face(x, y, w, h, img, gray, confidence_text):
    # Check if face has already been processed
    if is_face_processed(x, y, w, h):
        return

    # Add face to processed faces
    processed_faces.append((x, y, w, h))

    # Blur the face region
    face_region = img[y:y + h, x:x + w]
    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
    img[y:y + h, x:x + w] = blurred_face

    # Display confidence on the image
    cv2.putText(img, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Detect and blur eyes within frontal faces
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        eye_region = roi_color[ey:ey + eh, ex:ex + ew]
        blurred_eye = cv2.GaussianBlur(eye_region, (49, 49), 30)
        roi_color[ey:ey + eh, ex:ex + ew] = blurred_eye


for x, y, w, h, confidence_text in all_faces:
    process_face(x, y, w, h, img, gray, confidence_text)

# save_path = 'C:\\School\\599\\ProcessedImages\\test3.jpg'  # Change the path and filename as required
# cv2.imwrite(save_path, img)

# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Time taken to process the image: {elapsed_time:.2f} seconds")

# Display an image in a window
cv2.imshow('img', img)

# Wait for Esc key to stop
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

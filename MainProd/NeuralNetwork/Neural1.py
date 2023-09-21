import cv2
from mtcnn import MTCNN

# Initialize the MTCNN face detector
detector = MTCNN()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Detect faces in the frame using MTCNN
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Apply blur to the face region
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)

        # Replace the original face with the blurred face
        frame[y:y + h, x:x + w] = blurred_face

    # Display the processed frame
    cv2.imshow('Face Blurring', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
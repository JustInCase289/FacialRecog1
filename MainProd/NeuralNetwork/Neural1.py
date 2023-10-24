import cv2
from mtcnn import MTCNN
import time


def display_blurred_faces(filename):
    # Load image using OpenCV
    image = cv2.imread('C:\\Users\\bisho\\PycharmProjects\\FacialRecog\\MainProd\\TestImages'
                       '\\StockImageGroup3.jpg')

    # Convert the image from BGR to RGB format (as MTCNN expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Start the timer
    start_time = time.time()

    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)

    # For each detected face
    for face in faces:
        # Get coordinates and dimensions of the bounding box
        x, y, width, height = face['box']

        # Get the confidence score
        confidence = face['confidence']

        # Extract the face from the image
        face_image = image[y:y + height, x:x + width]
        # Blur the face
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
        # Replace the original face with the blurred version
        image[y:y + height, x:x + width] = face_image

        # Overlay the confidence score on the image
        confidence_text = f"{confidence:.2f}"
        cv2.putText(image, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # End the timer
    end_time = time.time()

    # Display the resulting image
    cv2.imshow('Blurred Faces', image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the window

    # Return the elapsed time
    return end_time - start_time


# Example usage
elapsed_time = display_blurred_faces('C:\\Users\\bisho\\PycharmProjects\\FacialRecog\\MainProd\\TestImages'
                                     '\\StockImageGroup3.jpg')
print(f"Time taken to process the image: {elapsed_time:.2f} seconds")

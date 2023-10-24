import cv2
from mtcnn import MTCNN


def display_blurred_faces(filename):
    # Load image using OpenCV
    image = cv2.imread("C:\\Users\\bisho\\PycharmProjects\\FacialRecog\\MainProd\\TestImages\\StockImageGroup4.jpg")

    # Convert the image from BGR to RGB format (as MTCNN expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN detector
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)

    # For each detected face
    for face in faces:
        # Get coordinates and dimensions of the bounding box
        x, y, width, height = face['box']
        # Extract the face from the image
        face_image = image[y:y + height, x:x + width]
        # Blur the face
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
        # Replace the original face with the blurred version
        image[y:y + height, x:x + width] = face_image

    # Display the resulting image
    cv2.imshow('Blurred Faces', image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the window


# Example usage
display_blurred_faces('input_image.jpg')

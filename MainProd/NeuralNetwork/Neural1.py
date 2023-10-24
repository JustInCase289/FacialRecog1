import cv2
from mtcnn import MTCNN
import time
import psutil


def get_resource_usage():
    cpu_usage = psutil.cpu_percent(interval=.1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    return cpu_usage, memory_usage


# Before running the main function
cpu_before, mem_before = get_resource_usage()


def display_blurred_faces(filename):
    # Load image using OpenCV
    image = cv2.imread(image_path)

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

    # Measure resource usage after processing the image but before displaying it
    cpu_after, mem_after = get_resource_usage()

    # Display the resulting image
    cv2.imshow('Blurred Faces', image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the window

    # Return the elapsed time
    return end_time - start_time, cpu_after, mem_after


# Single image path variable
image_path = 'C:\\Users\\bisho\\PycharmProjects\\FacialRecog\\MainProd\\TestImages\\StockImageGroup5.jpg'

# Example usage
elapsed_time, cpu_after, mem_after = display_blurred_faces(image_path)

print(f"Time taken to process the image: {elapsed_time:.2f} seconds")
print(f"CPU usage before: {cpu_before}% | After: {cpu_after}%")
print(f"Memory usage before: {mem_before}% | After: {mem_after}%")

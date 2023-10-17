import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True).eval()

# Load image
image = cv2.imread('C:\\School\\599\\StockImages\\StockImageGroup3.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to PyTorch tensor
tensor_img = F.to_tensor(image_rgb).unsqueeze(0)

# Get predictions
with torch.no_grad():
    prediction = model(tensor_img)

# Extract bounding boxes
boxes = prediction[0]['boxes']

for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    # Extract the face from the image
    face = image[y1:y2, x1:x2]
    # Blur the face
    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
    # Replace original face with blurred one
    image[y1:y2, x1:x2] = blurred_face

# Save or display the image
cv2.imshow('output.jpg', image)

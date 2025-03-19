import cv2
import numpy as np

# Function for preprocessing for fingerprint classification
def preprocess_image_fc(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    # Apply any preprocessing steps like resizing and thresholding
    image_resized = cv2.resize(image, (32, 32))  # Resizing to 32x32 pixels
    image_normalized = image_resized / 255.0  # Normalize the image to [0, 1]
    return image_normalized

# Function for general image preprocessing (e.g., equalization, resizing)
def preprocessing_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None
    # Histogram equalization to improve contrast
    equalized_image = cv2.equalizeHist(image)
    return image, equalized_image

# ultralytics/yolo/data/preprocessing.py

import cv2
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

def remove_hair(image):
    """
    Removes hair from skin lesion images using morphological operations.
    
    Args:
        image (np.ndarray): Input image in BGR format.
        
    Returns:
        np.ndarray: Image with hair removed.
    """
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.GaussianBlur(blackhat, (1, 1), 0)
    _, threshold = cv2.threshold(enhanced, 10, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    LOGGER.debug("Hair removed from image.")
    return inpainted_image

def apply_clahe(image):
    """
    Enhances the contrast of the image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image (np.ndarray): Input image in BGR format.
        
    Returns:
        np.ndarray: Image with enhanced contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    LOGGER.debug("CLAHE applied to image.")
    return enhanced_image

def color_normalization(image):
    """
    Normalizes the color channels of the image to have zero mean and unit variance.
    
    Args:
        image (np.ndarray): Input image in BGR format.
        
    Returns:
        np.ndarray: Color-normalized image.
    """
    mean_r = np.mean(image[:, :, 2])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 0])
    mean_gray = (mean_r + mean_g + mean_b) / 3.0
    scale_r = mean_gray / mean_r if mean_r != 0 else 1.0
    scale_g = mean_gray / mean_g if mean_g != 0 else 1.0
    scale_b = mean_gray / mean_b if mean_b != 0 else 1.0
    normalized_image = image.copy()
    normalized_image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)
    normalized_image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
    normalized_image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
    LOGGER.debug("Color normalization applied to image.")
    return normalized_image.astype(np.uint8)

def edge_enhancement(image):
    """
    Enhances the edges of the image using the Sobel operator and blends them with the original image.
    
    Args:
        image (np.ndarray): Input image in BGR format.
        
    Returns:
        np.ndarray: Image with enhanced edges.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    sobel_combined = sobel_combined.astype(np.uint8)
    enhanced_edges = cv2.merge((sobel_combined, sobel_combined, sobel_combined))
     LOGGER.debug("Edge enhancement applied to image.")
    return cv2.addWeighted(image, 0.9, enhanced_edges, 0.1, 0)

def custom_preprocessing(image):
    """
    Applies all custom preprocessing steps to the input image.
    
    Args:
        image (np.ndarray): Input image in BGR format.
        
    Returns:
        np.ndarray: Preprocessed image.
    """
    image = remove_hair(image)
    image = apply_clahe(image)
    image = color_normalization(image)
    image = edge_enhancement(image)
    LOGGER.debug("Custom preprocessing completed.")
    return image


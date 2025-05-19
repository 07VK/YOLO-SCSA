import cv2
import numpy as np

def remove_hair(image):
    """Remove hair and other artifacts using morphological operations and inpainting."""
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.GaussianBlur(blackhat, (1, 1), 0)
    _, threshold = cv2.threshold(enhanced, 10, 255, cv2.THRESH_BINARY)
    inpainted_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return inpainted_image

def apply_clahe(image):
    """Apply CLAHE to enhance local contrast in the L channel of LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def color_normalization(image):
    """Normalize color channels to reduce variation due to illumination/camera."""
    mean_r = np.mean(image[:, :, 2])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 0])
    mean_gray = (mean_r + mean_g + mean_b) / 3.0
    scale_r = mean_gray / (mean_r + 1e-6)
    scale_g = mean_gray / (mean_g + 1e-6)
    scale_b = mean_gray / (mean_b + 1e-6)

    normalized_image = image.copy()
    normalized_image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)
    normalized_image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
    normalized_image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
    return normalized_image.astype(np.uint8)

def edge_enhancement(image):
    """Enhance edges using Sobel gradients and merge with the original image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    sobel_combined = sobel_combined.astype(np.uint8)
    enhanced_edges = cv2.merge([sobel_combined]*3)
    return cv2.addWeighted(image, 0.9, enhanced_edges, 0.1, 0)

def custom_preprocessing(image):
    """Run all preprocessing steps in sequence."""
    img = remove_hair(image)
    img = apply_clahe(img)
    img = color_normalization(img)
    img = edge_enhancement(img)
    return img

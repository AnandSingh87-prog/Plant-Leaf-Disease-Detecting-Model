import cv2
import numpy as np

def is_leaf_image(image_path):
    """
    Validates if an image is likely a plant leaf using 4 feature-based heuristics:
    1. Color (Green Factor)
    2. Shape (Contour Analysis)
    3. Texture (Laplacian Variance)
    4. Edge Density (Canny Edge Ratio)
    """
    # 1. Load image
    image = cv2.imread(image_path)
    if image is None:
        return False, "Unable to read image."

    # Convert to different formats
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    conditions_met = 0
    total_conditions = 4
    
    # --- Feature 1: Color (Green Presence) ---
    # Define range for green (including yellowish-green)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
    
    # Threshold: At least 10% of image should be green-ish
    if green_ratio > 0.10:
        conditions_met += 1

    # --- Feature 2: Shape & Contour Analysis ---
    # Find contours from the green mask (or grayscale)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_found = False
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 1000: # Ignore tiny noise
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                # Circularity: 4*pi*Area / Perimeter^2
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                
                # Convex Hull for Solidity
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Leaves are organic: high solidity, moderate circularity
                # Not a perfect circle (circularity < 0.85)
                if solidity > 0.5 and circularity < 0.85:
                    shape_found = True

    if shape_found:
        conditions_met += 1

    # --- Feature 3: Texture Analysis (Laplacian Variance) ---
    # Leaves have veins and textures, giving a certain variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Threshold: Too smooth (low variance) or too noisy (very high)
    # Walls/Sky usually < 50. Natural texture usually 100 - 1500.
    if 50 < lap_var < 5000:
        conditions_met += 1

    # --- Feature 4: Edge Density ---
    # Detect edges
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    
    # Threshold: Leaves have moderate edge density due to veins
    # Typically 0.01 to 0.10. Artificial textures or noise > 0.2
    if 0.01 < edge_ratio < 0.15:
        conditions_met += 1

    # Final Decision: Decision logic (3 out of 4)
    is_valid = conditions_met >= 3
    
    return is_valid, "Valid leaf image" if is_valid else "Invalid image. Please upload a clear plant leaf."

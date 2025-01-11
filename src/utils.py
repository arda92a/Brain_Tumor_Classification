import cv2
import numpy as np
import os
from scipy import stats

def convert_to_grayscale(image):
    """
    Converts a single input image from RGB to grayscale.
    
    Args:
        image (numpy.ndarray): Input image in RGB format.
        
    Returns:
        numpy.ndarray: Grayscale image.
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale_image

def apply_threshold(image, threshold_value=70):
    """
    Applies thresholding to a single input image to create a binary image.
    
    Args:
        image (numpy.ndarray): Input image in grayscale format.
        threshold_value (int): Value to use as the threshold (default is 128).
        
    Returns:
        numpy.ndarray: Binary (thresholded) image.
    """

    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image

def apply_dilation(image, kernel_size=(6, 6), iterations=1):
    """
    Applies dilation to the input binary image.
    
    Args:
        image (numpy.ndarray): Input binary image.
        kernel_size (tuple): Size of the structuring element (default is (6, 6)).
        iterations (int): Number of times dilation is applied (default is 1).
        
    Returns:
        numpy.ndarray: Dilated image.
    """
    kernel = np.ones(kernel_size, np.uint8)  # Structuring element
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_image

def detect_and_draw_keypoints(image):
    """
    Detects keypoints in the input image using ORB and draws them.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        
    Returns:
        numpy.ndarray: Image with keypoints drawn.
    """
    orb = cv2.ORB_create()

    kp = orb.detect(image, None)

    kp, des = orb.compute(image, kp)

    img_with_keypoints = cv2.drawKeypoints(image, kp,image)

    return img_with_keypoints

def adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(16, 16)):

    """
    Applies adaptive histogram equalization (CLAHE) to a grayscale image to enhance contrast.
    
    Args:
        image (numpy.ndarray): Input grayscale image.
        clip_limit (float): Threshold for contrast limiting (default is 2.0).
        grid_size (tuple): Size of the grid for the histogram equalization (default is (8, 8)).
        
    Returns:
        numpy.ndarray: Adaptive histogram-equalized image.
    """

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    

    equalized_image = clahe.apply(image)
    
    return equalized_image

def apply_histogram_equalization_to_folder(folder_path, output_folder):
    """
    Applies histogram equalization to all images in a specified folder and saves the results.
    
    Args:
        folder_path (str): Path to the folder containing images.
        output_folder (str): Path to the folder to save equalized images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
          
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized_image = adaptive_histogram_equalization(gray_image)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, equalized_image)

def pixel_intensity_stats(data_dir):
    """
    Calculates pixel intensity statistics (mean, median, standard deviation, minimum, maximum) 
    for images in the 'Brain Tumor' and 'Healthy' categories and prints the results.
    
    Args:
        data_dir (str): Path to the directory containing the subdirectories 'Brain Tumor' and 'Healthy'.
                        Each subdirectory should contain image files to analyze.
    
    Returns:
        None: This function prints the statistics for each category directly to the console.
    """
    categories = ["Brain Tumor", "Healthy"]
    valid_extensions = ('.jpeg', '.jpg', '.png', '.tiff')
    
    stats = {}  
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        pixel_values = []

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if file_path.lower().endswith(valid_extensions):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  
                if img is not None:
                    pixel_values.extend(img.ravel()) 
        
        if pixel_values:
            pixel_values = np.array(pixel_values)
            stats[category] = {
                "Mean": np.mean(pixel_values),
                "Median": np.median(pixel_values),
                "Std Deviation": np.std(pixel_values),
                "Min": np.min(pixel_values),
                "Max": np.max(pixel_values)
            }
        else:
            stats[category] = "No valid images found!"
    
    for category, category_stats in stats.items():
        print(f"\nCategory: {category}")
        if isinstance(category_stats, dict):
            for stat, value in category_stats.items():
                print(f"{stat}: {value:.2f}")
        else:
            print(category_stats)

def analyze_pixel_values(data_dir):
    """
    Analyzes the pixel values of images in the 'Brain Tumor' and 'Healthy' categories to check for normality 
    and perform statistical tests to determine if there is a significant difference between the two categories.
    
    Args:
        data_dir (str): Path to the directory containing subdirectories 'Brain Tumor' and 'Healthy', 
                         each containing image files for analysis.
    
    Returns:
        None: This function prints the results of normality tests and statistical tests to the console.
    """
    categories = ["Brain Tumor", "Healthy"]
    valid_extensions = ('.jpeg', '.jpg', '.png', '.tiff')
    
    pixel_values_tumor = []
    pixel_values_healthy = []

    for category in categories:
        category_path = os.path.join(data_dir, category)
        
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            if file_path.lower().endswith(valid_extensions):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  
                if img is not None:
                    pixel_values = img.ravel()  
                    mean_pixel_value = pixel_values.mean()  
                    if category == "Brain Tumor":
                        pixel_values_tumor.append(mean_pixel_value)
                    else:
                        pixel_values_healthy.append(mean_pixel_value)

    tumor_normality = stats.shapiro(pixel_values_tumor)[1] > 0.05
    healthy_normality = stats.shapiro(pixel_values_healthy)[1] > 0.05

    print(f"Brain Tumor Normality Test: {'Passed' if tumor_normality else 'Failed'}")
    print(f"Healthy Normality Test: {'Passed' if healthy_normality else 'Failed'}")

    if tumor_normality and healthy_normality:
        t_stat, p_value = stats.ttest_ind(pixel_values_tumor, pixel_values_healthy)
        print("\nIndependent t-Test Results:")
        print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    else:
        u_stat, p_value = stats.mannwhitneyu(pixel_values_tumor, pixel_values_healthy)
        print("\nMann-Whitney U Test Results:")
        print(f"U-statistic: {u_stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("\nThere is a statistically significant difference between the two categories.")
    else:
        print("\nThere is no statistically significant difference between the two categories.")



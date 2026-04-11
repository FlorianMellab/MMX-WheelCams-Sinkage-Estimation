import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import cv2
import os
import scipy.io
from skimage.filters import threshold_otsu

"""
Wheel edge detection for the rear wheel.

This script is designed to detect the edges of a wheel in images captured by a rear-facing camera.

The script requires:
    - The raw images of the rear wheel (in .tiff or .png format)
    
The output includes:
    - Detected wheel contour plotted on the original image for visual verification
    - Coordinates of the detected contour saved in a .mat file for subsequent ellipse-fitting analysis (with user confirmation)

The pipeline consists of the following steps:
- Read the image and convert it to grayscale if necessary.
- Define a confidence region where the wheel is expected to be located and compute the average pixel intensity in this region.
- Perform brightness-based segmentation by keeping pixels within an adaptive intensity range defined by the average pixel value
- Apply morphological transformation to fill small gaps in the detected shapes and to remove noise inside the foreground object.
- Use Canny edge detection to extract edges from the refined mask.
- Find continuous contours in the edge map
- Select the contour with the largest minimum-enclosing-circle radius
  (largest radius among circles defined as the smallest circle that encloses all the detected edges) as the wheel boundary.
- Export the coordinates of the detected contour to a .mat file for subsequent ellipse-fitting analysis.

The parameters for the confidence region, brightness segmentation thresholds, morphological kernel sizes, and Canny edge detection 
thresholds were adjusted for optimal performance with rear wheel images.
"""

def txt_to_numpy_array(file_path):
    
    """Reads a text file containing pixel values and converts it to a normalized numpy array (0-255)."""

    with open(file_path, 'r') as f:
        pixel_data = [list(map(int, line.strip().split())) for line in f if line.strip()]
    
    array = np.array(pixel_data, dtype=np.int32)  # Use a safe dtype first

    # Normalize to 0–255
    norm_array = 255 * (array - np.min(array)) / (np.max(array) - np.min(array))
    norm_array = norm_array.astype(np.uint8)
    return norm_array


def read_image(image_path):
    """Read image as an array from an absolute path"""
    image = skimage.io.imread(image_path)
    # print("Image dimension: ", image.shape)
    return image

def convert_2_grey(image):
    """Convert an RGB image to greyscale"""
    image_grey = skimage.color.rgb2gray(image)
    return image_grey

def visualize(image, title=None):
    """Visualize a color or binary image with scikit image library"""
    plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap='gray', origin='upper')
    plt.title(title)
    # plt.axis('off')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()   
    return

def extract_wheel_confidence_area(image_grey, m1, n1, a1, b1):
    
    """
    Extracts the average pixel value inside a given confidence area from a binary image
    # (n1, m1) is top left corner of confidence area
    # b1, a1 are the x and y sizes of confidence area

    """
    wheel_confidence_area = image_grey[n1:n1+b1, m1:m1+a1]
    average_value = np.mean(wheel_confidence_area)
    return wheel_confidence_area, average_value

def segment_image(image, average_value, Pa, Pd):

    """
    Segments the image based on thresholding with average value.
    Pa and Pd are the parameters that define the thresholding range.
    Pixels are kept if their value is between average_value * Pa and average_value * Pd.
    """
    min_value, max_value = average_value * Pa, average_value * Pd
    segmented_image = np.where((image >= min_value) & (image <= max_value), 1, 0).astype(np.uint8)
    plt.figure(figsize=(10, 6))
    plt.imshow(segmented_image, cmap='gray', origin='upper')
    plt.title("Segmented image")
    # plt.axis('off')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()   
    return segmented_image

def segment_image_Otsu(image):
    """
    Segments the image based on Otsu thresholding with global thresholding, by maximising the variance between the 2 pixel groups. 

    """
    thresh = threshold_otsu(image)
    segmented_image = image > thresh

    # Optional visualization of the segmented image
    plt.figure(figsize=(10, 6))
    plt.imshow(segmented_image, cmap='gray', origin='upper')
    plt.title("Segmented image with Otsu")
    plt.axis('off')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()   
    segmented_image = segmented_image.astype(int)
    print(np.max(segmented_image))

    return segmented_image

def perform_morphological_transform(segmented_image, kernel_closing, kernel_opening):
    """Apply morphological transformation to improve the quality of the segmentation mask 
    by filling holes in the shape (closing) and removing protrusions (opening)"""

    # Convert segmented_image to uint8 for OpenCV functions
    segmented_image = segmented_image.astype(np.uint8)

    # Apply morphological closing followed by opening
    closing = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel_closing, iterations=2)
    opening  = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening, iterations=2)
    mask = opening

    # # Optional visualization of the mask after morphological transform
    plt.figure(figsize=(10, 6))
    plt.imshow(mask, cmap='gray')
    plt.title("Segmented image with morphological transform")
    # plt.axis('off')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
      
    return mask

def detect_edges(segmented_image, canny_lower, canny_upper):
    """Apply Guassian blurr and perform Canny edge detection on the segmented image"""

    # Apply Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(segmented_image, (5, 5), 1)  # (Kernel size, Standard deviation)

    # Canny edge detection requires an 8-bit single-channel image, so we need to convert the blurred image to uint8
    blurred_255 = (blurred * 255).astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_255, canny_lower, canny_upper)

    # edges[:, 0] = 255     # Closing the shape is not necessary anymore after switching to min eclosing circle method

    # # Optional visualization of the detected edges
    # plt.figure(figsize=(6, 6))
    # plt.imshow(edges, cmap='gray')
    # plt.title("Edge Detection (Canny)")
    # plt.axis('off') 
    # plt.show()    

    return edges 

def find_contours(edges):

    """
    Find continuous contours given edge map and filter based on enclosed area. 
    Note that if the contour is not closed, it will attempt to calculate an area by polygon approximation
    """

    # Find contours in the edge map. Hierarchy corresponds to the contour hierarchy based on the nesting of contours (e.g., holes within objects).
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found, if not, raise an error
    if len(contours) == 0:
        raise ValueError('No contours found!')

    # Selecting the contour to be used among all those found using different criteria
    wheel_contour = max(contours, key=lambda cnt: cv2.minEnclosingCircle(cnt)[1])   #Allows filtering by maximizing the radius of minimizing enclosing circle
    # wheel_contour = max(contours, key=cv2.contourArea)                            #Allows filtering by maximizing the enclosedarea, less robust
    # wheel_contour = max(contours, key=lambda cnt: cv2.arcLength(cnt, False))      #Allows filtering by maximizing the contour length, less robust

    # Calculate the center and radius of the minimum enclosing circle for the selected contour.
    (x,y),radius = cv2.minEnclosingCircle(wheel_contour)
    print("Radius of enclosing circle", int(radius))

    #Check that the detected contour is a real candidate

    if cv2.contourArea(wheel_contour) > 100000:  # Area threshold of 100000 was determined empirically based on visual observation of the contour on the images.
        
        # print("Wheel area correct", str(cv2.contourArea(wheel_contour)))

        wheel_contour = np.array(wheel_contour)         # Convert contour to numpy array for plotting
        wheel_contour = np.squeeze(wheel_contour)       # Remove redundant dimensions for easier plotting

        #Plotting the detected contour on top of the mask for visual verification
        plt.figure(figsize=(10, 6))
        plt.imshow(mask, cmap="gray", origin='upper')
        plt.plot(wheel_contour[:, 0], wheel_contour[:, 1], linewidth=2, color="red")
        # plt.axis("off")
        plt.title("Wheel Optimal Contour")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    else:

        # print("Wheel area too small!", str(cv2.contourArea(wheel_contour)))

        wheel_contour = np.array(wheel_contour)             # Convert contour to numpy array for plotting
        wheel_contour = np.squeeze(wheel_contour)           # Remove redundant dimensions for easier plotting

        #Plotting the detected contour on top of the mask for visual verification
        plt.figure(figsize=(10, 6))
        plt.imshow(image, cmap="gray", origin='upper')
        plt.plot(wheel_contour[:, 0], wheel_contour[:, 1], linewidth=2, color="red")
        # plt.axis("off")
        plt.title("Wheel Contour - Unvalid candidate, too small area")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    return wheel_contour

if __name__ == "__main__":

    #####################################
    # CHOOSE THE IMAGE(S) TO BE PROCESSED
    #####################################

    folder_path = None

    list_images = ["images/Raw images JAXA/R_0.5317.jpg"]           #Back wheel cam
    # list_images = ["images/low_sinkage_front.png"]                #Front wheel cam

    # # Path to raw images of the rear wheel for each simulant while the wheel is rotating
    # folder_path = "images/Raw images JAXA"        # JAXA simulant 
    # folder_path = "images/Raw images moon"        # Moon simulant

    if folder_path is not None:
        list_images = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg')):
                image_path = os.path.join(folder_path, filename)
                list_images.append(image_path)

    for image_path in list_images:

        ############################################
        # EXTRACTING THE IMAGES / AREA TO BE STUDIED
        ############################################

        # Depending on file type use read_image or txt to numpy array
        if image_path.lower().endswith('.txt'):         # Add more extensions if you have other types of raw data files that need to be read differently
            image = txt_to_numpy_array(image_path)
        else:
            image = read_image(image_path) 

        # # Optional cropping of the image to focus on the area where the wheel is expected to be located, or for piecewise segmentation. 
        # image = image[:, :1100]       # Limit the analysis to the left half of the image
        # image = image[:, 700:]        # Limit the analysis to the right half of the image

        visualize(image, title="Raw image")

        #######################################
        # PRE-PROCESSING THE IMAGE FOR ANALYSIS
        #######################################

        if len(image.shape) == 3:
            image_grey = convert_2_grey(image)
        else:
            image_grey = image


        ####################################################
        # MODEL PARAMETERS - ADJUSTED FOR REAR WHEEL IMAGES
        ####################################################

        # Confidence area parameters where the wheel is expected to be located (its top left corner position (m1, n1) and size (a1, b1)) 
        m1, n1, a1, b1 = 0, 400, 200, 200       # Model ajusted for full image
        conf_area = [m1, n1, a1, b1]

        # Brightness segmentation parameters (Pa and Pd) to define the thresholding range.
        Pa, Pd = 0.35, 1.85         # Optimized for full images of the rear wheel

        # Defining kernel sizes for morphological closing and opening operations. These influence the extent of gap filling and noise removal
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))  # Elliptical kernel keeps more natural shape
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60,60))  # Elliptical kernel keeps more natural shape

        # Defining the Canny edge detection thresholds. These parameters control the sensitivity of edge detection.      
        canny_lower = 220
        canny_upper = 255

        ######################
        # SEGMENTATION PROCESS
        ######################        

        wheel_confidence_area, average_value = extract_wheel_confidence_area(image_grey, m1, n1, a1, b1)

        segmented_image = segment_image(image_grey, average_value, Pa, Pd)

        #########################
        # MORPHOLOGICAL TRANSFORM 
        #########################         

        mask = perform_morphological_transform(segmented_image, kernel_closing=kernel_closing, kernel_opening=kernel_opening) 

        #######################################
        # EDGE DETECTION AND CONTOUR EXTRACTION
        ########################################

        edges = detect_edges(mask, canny_lower=canny_lower, canny_upper=canny_upper)

        wheel_contour = find_contours(edges)

        ##############################################
        # SAVING THE CONTOUR AND EXPORTING THE RESULTS
        ###############################################
        """ The coordinates of the detected contour are saved in a .mat file (with user-provided name) for subsequent ellipse-fitting analysis"""

        # Split into x and y column vectors
        x_points = wheel_contour[:, 0].reshape(-1, 1)
        y_points = wheel_contour[:, 1].reshape(-1, 1)

        # Prompt user to save the results
        user_input = input("Do you want to save the contour? (Y/N): ").strip().upper()
        if user_input == 'Y':
            filename = input("Enter a filename: ").strip()
            if filename == '':
                filename = 'wheel_edge'
            scipy.io.savemat(f'{filename}.mat', {
                'x_points': x_points,
                'y_points': y_points,
                'I': image,
                'wheel_confidence_area': conf_area,
                'Pa': Pa,
                'Pd': Pd,
                'kernel_closing': kernel_closing,
                'kernel_opening': kernel_opening,
                'canny_min': canny_lower,
                'canny_max': canny_upper             })
            print(f"Saved as {filename}.mat")
        else:
            print("Result not saved.")

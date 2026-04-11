import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import cv2
import os
import scipy.io
from skimage.filters import threshold_otsu

"""
Wheel edge detection for the front wheel.

This script is designed to detect the edges of a wheel in images captured by IDEFIX Front WheelCams.

The script requires:
    - The raw images of the front wheel (in .tiff or .png format)
    
The output includes:
    - Detected wheel contour plotted on the original image for visual verification
    - Coordinates of the detected contour saved in a .mat file for subsequent ellipse-fitting analysis (with user confirmation)

The pipeline consists of the following steps:
- Read the image and convert it to grayscale if necessary.
- Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast of the image.
- Define a confidence region where the wheel is expected to be located and compute the average pixel intensity in this region.
  Here, piecewise segmentation can be applied (separating the left and right halves of the wheel) to account for different 
  lighting conditions across the wheel.
- Perform brightness-based segmentation by keeping pixels within an adaptive intensity range defined by the average pixel value
- Load a separate texture-based mask (if available) and combine it with the brightness-based segmentation mask using 
  a logical OR operation to create a more robust foreground mask.
- Apply morphological transformation to fill small gaps in the detected shapes, and to remove noise and the wheel spokes.
- Use Canny edge detection to extract edges from the refined mask.
- Find continuous contours in the edge map
- Select the contour with the largest minimum-enclosing-circle radius 
  (largest radius among circles defined as the smallest circle that encloses all the detected edges) as the wheel boundary.
- Export the coordinates of the detected contour to a .mat file for subsequent ellipse-fitting analysis.

The parameters for the confidence region, brightness segmentation thresholds, morphological kernel sizes, and Canny edge detection thresholds 
were adjusted for optimal performance with front wheel images.
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

def preprocess(image):
    """Apply CLAHE to enhance the contrast of the image"""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(10,10))
    cl1 = clahe.apply(image)
    return cl1

def extract_wheel_confidence_area(image_grey, m1, n1, a1, b1):

    """
    Extracts the average pixel value inside a given confidence area from a binary image
    # (n1, m1) is top left corner of confidence area
    # b1, a1 are the x and y sizes of confidence area

    """
    wheel_confidence_area = image_grey[n1:n1+b1, m1:m1+a1]
    average_value = np.mean(wheel_confidence_area)
    print("The average intensity in the confidence area is:", average_value)
    return wheel_confidence_area, average_value

def segment_image(image, average_value, Pa, Pd):

    """
    Segments the image based on thresholding with average value.
    Pa and Pd are the parameters that define the thresholding range.
    Pixels are kept if their value is between average_value * Pa and average_value * Pd.
    """
    min_value, max_value = average_value * Pa, average_value * Pd
    segmented_image = np.where((image >= min_value) & (image <= max_value), 1, 0).astype(np.uint8)
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
    plt.title("Segmented image")
    plt.axis('off')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()   
    segmented_image = segmented_image.astype(int)
    print(np.max(segmented_image))

    return segmented_image

def perform_morphological_transform(segmented_image, kernel_closing, kernel_opening):
    """Apply morphological transformation operation to fill small gaps in the detected shapes and opening to remove noise and the wheel spokes."""

    # Convert segmented_image to uint8 for OpenCV functions
    segmented_image = segmented_image.astype(np.uint8)

    # Apply morphological closing followed by opening
    closing = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel_closing, iterations=1)
    opening  = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_opening, iterations=1)
    mask = opening

    # # Optional visualization of the mask after morphological transform
    # plt.figure(figsize=(10, 6))
    # plt.imshow(mask, cmap='gray')
    # # plt.title("Segmented image with morphological transform")
    # # plt.axis('off')
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()
  
    return mask

def detect_edges(segmented_image, canny_lower, canny_upper):

    """Apply Guassian blurr and perform Canny edge detection on the segmented image"""

    # Apply Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(segmented_image, (5, 5), 1)  # (Kernel size, Standard deviation)

    # Canny edge detection requires an 8-bit single-channel image, so we need to convert the blurred image to uint8
    blurred_255 = (blurred * 255).astype(np.uint8)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred_255, canny_lower, canny_upper)

    # edges[:, 0] = 255 # Closing the shape is not necessary anymore after switching to min eclosing circle method

    # # Optional visualization of the detected edges
    # plt.figure(figsize=(6, 6))
    # plt.imshow(edges, cmap='gray')
    # plt.title("Edge Detection (Canny)")
    # plt.axis('off') 
    # plt.show()
         
    return edges 

def find_contours(edges, mask):

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
    # wheel_contour = max(contours, key=cv2.contourArea)                            #Allows filtering by maximizing the enclosed area, less robust
    # wheel_contour = max(contours, key=lambda cnt: cv2.arcLength(cnt, False))      #Allows filtering by maximizing the contour length, less robust

    # Calculate the center and radius of the minimum enclosing circle (that encloses all the edtected edges) for the selected contour.
    (x,y),radius = cv2.minEnclosingCircle(wheel_contour)
    print("Radius of enclosing circle", int(radius))

    #Check that the detected contour is a real candidate
    if cv2.contourArea(wheel_contour) > 100000:                 # Area hreshold of 100000 was determined empirically based on visual observation of the contour on the images.

        # print("Wheel area correct", str(cv2.contourArea(wheel_contour)))

        wheel_contour = np.array(wheel_contour)                 # Convert contour to numpy array for plotting
        wheel_contour = np.squeeze(wheel_contour)               # Remove redundant dimensions for easier plotting

        #Plotting the detected contour on top of the mask for visual verification
        plt.figure(figsize=(10, 6))
        plt.imshow(mask, cmap="gray", origin='upper')
        plt.plot(wheel_contour[:, 0], wheel_contour[:, 1], linewidth=2, color="red")
        # plt.axis("off")
        plt.title("Detected wheel contour")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    else:
        # print("Wheel area too small!", str(cv2.contourArea(wheel_contour)))

        wheel_contour = np.array(wheel_contour)             # Convert contour to numpy array for plotting
        wheel_contour = np.squeeze(wheel_contour)           # Remove redundant dimensions for easier plotting

        #Plotting the detected contour on top of the mask for visual verification
        plt.figure(figsize=(10, 6))
        plt.imshow(mask, cmap="gray", origin='upper')
        plt.plot(wheel_contour[:, 0], wheel_contour[:, 1], linewidth=2, color="red")
        # plt.axis("off")
        plt.title("Detected wheel contour - Unvalid candidate, too small area")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    return wheel_contour

if __name__ == "__main__":

    #####################################
    # CHOOSE THE IMAGE(S) TO BE PROCESSED
    #####################################

    folder_path = None

    # # Back and front wheel tests
    # list_images = ["images/test_image_GRAVEL.jpg"]        #Back wheel cam
    # list_images = ["images/high_sinkage_front.png"]       #Front wheel cam

    # # SNR images from Alice (varying illumination conditions)
    # list_images = ["images/MAE_SNR/Jaxa/Art3D_Ground_Jaxa_100ms_LED30_S/F_1001_0.31396.tiff"]     #30s SNR Front JAXA, 
    # list_images = ["images/MAE_SNR/Jaxa/Art3D_Ground_Jaxa_100ms_LED50_S/F_1001_0.36155.tiff"]     #50s SNR Front JAXA
    # list_images = ["images/MAE_SNR/Jaxa/Art3D_Ground_Jaxa_100ms_LED55_S2/F_1001_0.33518.tiff"]    #55s SNR Front JAXA
    
    # # Donatien ground truth mask
    # list_images = ["Donatien segmentation mask\PP_F_1154_57.3487.png"]

    # # Path to raw images of the rear wheel for each simulant while the wheel is rotating
    # folder_path = "images/Raw images JAXA"            # JAXA simulant
    # folder_path = "images/Raw images moon"            # Moon simulant

    # # Stress testing the algorithm with different SNR amd different simulants
    # folder_path = "images/SNR image bank/JAXA"
    # folder_path = "images/SNR image bank/Moon"
    # folder_path = "images/SNR image bank/DLR"

    # # Best SNR image from SNR test for each simulant
    list_images = ["images/SNR image bank/JAXA/JAXA_50s.tiff"] #50s SNR Front JAXA, 13.6% sat
    # list_images = ["images/SNR image bank/Moon/Moon_10s.tiff"] #10s SNR Front Moon, 3.1% sat
    # list_images = ["images/SNR image bank/DLR/DLR_100ms_LED5.tiff"] #100ms_LED4 Front DLR, 9.1% sat

    # # Testbench laser measurement validation 15012026
    # list_images = ["images/15012026/SmallGravel_SinkageTest/Front/F_11_1.0616.tiff"]
    # list_images = ["images/15012026/SmallGravel_SinkageTest_HigherSinkage/Front/F_11_0.047998.tiff"]
    
    # # Testbench laser measurement validation 18022026
    # list_images = ["images/18022026_ValidationSet/Extracted Data/Test_2_with_Sinkage_085939_2/images_front/cam_front_000000.png"]


    # Extracting all the images if they are stored in a folder and storing their paths in a list for processing.
    if folder_path is not None:
        list_images = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.tiff')):
                image_path = os.path.join(folder_path, filename)
                list_images.append(image_path)
            elif filename.lower().endswith(('.png')):
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

        # # Optional cropping of the image to focus on the area where the wheel is expected to be located or for piecewise segmentation. 
        # # Adjust the cropping parameters as needed based on the camera angle and field of view.
        image = image[:, :1100]       # Limit the analysis area to the left half of the image 
        # image = image[:, 700:]          # Limit the analysis area to the right half of the image
        # image = image[:, 100:1200]    # New angle once the testbed was modified

        visualize(image, title="Raw image")

        #######################################
        # PRE-PROCESSING THE IMAGE FOR ANALYSIS
        #######################################

        # If the image is in color, convert it to grayscale for further processing.
        if len(image.shape) == 3:
            image_grey = convert_2_grey(image)
        else:
            image_grey = image

        # Applying the CLAHE pre-processing step to the grayscale image
        image_clahe = preprocess(image_grey)
        visualize(image_clahe, title="CLAHE pre-processing")

        ####################################################
        # MODEL PARAMETERS - ADJUSTED FOR FRONT WHEEL IMAGES
        ####################################################

        # Confidence area parameters where the wheel is expected to be located (its top left corner position (m1, n1) and size (a1, b1)) 
        # m1, n1, a1, b1 = 200, 200, 200, 200         # Right half of the wheel (with piecewise segmentation) 
        m1, n1, a1, b1 = 400, 400, 200, 200       # Basic model for full image / Left half of the wheel (with piecewise segmentation) 
        # m1, n1, a1, b1 = 150, 0, 200, 200         # Basic model for full image, new camera angle
        conf_area = [m1, n1, a1, b1]

        # Brightness segmentation parameters (Pa and Pd) to define the thresholding range.
        Pa, Pd = 0.5, 1.5         # Basic model for full image / Left half of the wheel
        # Pa, Pd = 0.7, 1.2           # Optimized for right half (0.7,1.2)

        # Defining kernel sizes for morphological closing and opening operations. These influence the extent of gap filling and noise removal
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))  # Elliptical kernel keeps more natural shape
        kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))  # Elliptical kernel keeps more natural shape

        # Defining the Canny edge detection thresholds. These parameters control the sensitivity of edge detection.
        canny_lower = 225
        canny_upper = 255

        ######################
        # SEGMENTATION PROCESS
        ######################

        # Obtaining the confidence area and average pixel intensity based on the 
        wheel_confidence_area, average_value = extract_wheel_confidence_area(image_clahe, m1, n1, a1, b1)

        # Segmmenting the image using the adaptive brightness-based method
        segmented_image = segment_image(image_clahe, average_value, Pa, Pd) 

        # Visualize the segmented image before morphological transformation
        mask_brightness = segmented_image
        visualize(mask_brightness, title="Segmented image mask")

        ######################################
        # OPTIONAL: TEXTURE-BASED SEGMENTATION
        ######################################

        """ Texture-based segmentation was attempted but did not yeild conclusive results, so it is not included in the current pipeline """
        
        # # Loading a texture-based segmentation mask (if available). 
        # segmented_image = np.load("texture_mask.npy")

        # # Load the .mat file in Python. Choose the desired mask
        # mat_file = r"texture_mask_high_sinkage.mat"       # or try mat_file = r"saved segmentation masks/texture_mask_high_sinkage.mat"
        # mat_file = r"texture_mask_high_sinkage_2.mat"     # or try mat_file = r"saved segmentation masks/texture_mask_high_sinkage_2.mat"
        # mat_file = r"Donatien image mask.mat"    
        # mat_file = r"texture_mask_donatien.mat"           # or try mat_file = r"saved segmentation masks/texture_mask_donatien.mat"

        # data = scipy.io.loadmat(mat_file)
        # mask_texture = data['mask'].squeeze()

        # # Plotting the texture-based mask for visual verification
        # plt.figure(figsize=(10, 6))
        # plt.imshow(mask_texture, cmap='gray')
        # plt.title("Texture mask")
        # plt.axis('off')
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.show()  

        # # Combine both masks
        # combined_mask = mask_brightness | mask_texture    # Logical OR operation to combine the masks
        # plt.figure(figsize=(10, 6))
        # plt.imshow(combined_mask, cmap='gray')
        # plt.title("Combined mask")
        # plt.axis('off')
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.show()  

        #########################
        # MORPHOLOGICAL TRANSFORM 
        ######################### 

        # mask = perform_morphological_transform(mask_brightness, kernel_closing=kernel_closing, kernel_opening=kernel_opening)
        mask = mask_brightness

        # Padding the mask to ensure that contours touching the borders are properly detected
        padded_mask = np.pad(
        mask,
        pad_width=((5, 5), (5, 5)),
        mode='constant',
        constant_values=0
        ) 
        
        #######################################
        # EDGE DETECTION AND CONTOUR EXTRACTION
        ########################################

        edges = detect_edges(padded_mask, canny_lower=canny_lower, canny_upper=canny_upper)

        wheel_contour = find_contours(edges, mask=padded_mask)

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
                'canny_max': canny_upper,
                'mask': padded_mask             })
            print(f"Saved as {filename}.mat")
        else:
            print("Result not saved.")

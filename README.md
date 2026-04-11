# MMX-WheelCams-Sinkage-Estimation
Image processing pipeline to estimate sinkage from IDEFIX WheelCam images.
Contact details: Florian Mellab, florian.mellab@gmail.com

# Description of the pipeline:
1. edge_detection_front.py or edge_detection_rear.py (raw image .png or .tiff -> detected contour .mat)
2. ellipse_fit_new.py (detected contour .mat -> ellipse model .mat)
3. homography.py (ellipse model .mat -> sinkage estimation)

# Other scripts
compute_saturation.py: Compute saturation level of a raw image
ellipse_comparison_metric.py: Computes similarity metrics between 2 ellipse models
manual_sinkage_tool.py: Compute an ellipse fit from a raw image using manual inputs
plot_ellipse.py: Visualise an ellipse, its inlier and outlier points overlayed on a raw image

# System requirements: 
imageio==2.37.0
matlab==0.1
matplotlib==3.5.1
numpy==1.24.2
numpy==1.22.0
opencv_python==4.7.0.72
pandas==2.0.1
scikit_learn==1.2.2
scipy==1.7.3
skimage==0.0

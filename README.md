# MMX-WheelCams-Sinkage-Estimation

Image processing pipeline to estimate wheel sinkage from IDEFIX WheelCam images.

Contact details: Florian Mellab, florian.mellab@gmail.com

# Description of the pipeline:

1. edge_detection_front.py or edge_detection_rear.py (raw image .png or .tiff -> detected contour .mat)
   
3. ellipse_fit_new.py (detected contour .mat -> ellipse model .mat)
   
5. homography.py (ellipse model .mat -> sinkage estimation)

# Other scripts

compute_saturation.py: Compute saturation level of a raw image

ellipse_comparison_metric.py: Computes similarity metrics between 2 ellipse models

manual_sinkage_tool.py: Compute an ellipse fit from a raw image using manual inputs

plot_ellipse.py: Visualise an ellipse, its inlier and outlier points overlayed on a raw image

# System requirements: 
imageio

matlab

matplotlib

numpy

opencv-python

pandas

scikit-learn

scipy

scikit-image

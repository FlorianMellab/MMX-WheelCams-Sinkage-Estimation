import numpy as np
import pandas as pd

#This file is used to determine wheel sinkage from laser measurement, which will be used to validate the automatic sinkage detection algorithm

#Definitioan of variables
wheel_diameter = 214 #mm
laser_to_wheel_bottom = 50 #mm THIS IS AN APPROX VALUE, STILL NEED TO MEASURE IN TESTBED

def extract_measurements(file_name):
    file = pd.read_excel(file_name)
    laser_measurement = file.loc[0, "pos_Y"] 
    laser_regolith_distance = file.loc[0, "laser_regolith_Y"] 
    return laser_measurement, laser_regolith_distance

def compute_sinkage(laser_measurement, laser_to_wheel_bottom, laser_regolith_distance):
    sinkage = laser_measurement + laser_to_wheel_bottom - laser_regolith_distance
    return sinkage

if __name__ == "__main__":
    file_name = "images/15012026/ExcelSmallGravel_SinkageTest_HigherSinkage.xlsx"
    laser_measurement, laser_regolith_distance = extract_measurements(file_name)
    sinkage = compute_sinkage(laser_measurement=laser_measurement, laser_to_wheel_bottom=laser_to_wheel_bottom, laser_regolith_distance=laser_regolith_distance)
    print("Wheel sinkage [mm]: ", sinkage)

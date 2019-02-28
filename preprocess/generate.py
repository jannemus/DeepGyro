import numpy as np

# Load camera and IMU calibration information
import calibration as calib

import IO    # For reading and writing data
from utils import *
from visualize import plotBlurVectors

if __name__ == '__main__':

    inpath, outpath = IO.parseInputs()
    print("Input folder: %s" %inpath)
    
    ''' Read input data and calibration parameters '''
    
    # Read gyroscope measurements and timestamps
    gyr, tgyr = IO.readGyroscope(inpath)

    # Read image timestamps and exposure times
    tf, te = IO.readImageInfo(inpath)
    
    # Load calibration parameters from calibration.py
    scaling = calib.scaling # Downsample images?
    K = calib.K             # Camera intrinsics
    tr = calib.tr           # Camera readout time
    td = calib.td           # IMU-camera temporal offset
    Ri = calib.Ri           # IMU-to-camera rotation
    
    # If we downsample, we also need to scale intrinsics
    if scaling < 1.0:
        K = scaling*K
        K[2,2] = 1
        
    IO.createOutputFolders(outpath)
     
    ''' Temporal and spatial alignment of gyroscope and camera '''
    
    # Read the first image to get image dimensions
    img = IO.readImage(inpath, scaling, idx=0)
    height, width = img.shape[:2]
    
    dt = tr/height # Sampling interval
    gyr = alignSpatial(gyr, Ri)
    gyr, t, tf, te = alignTemporal(gyr, tgyr, tf, te, tr, td, dt)
    
    ''' Generate blur field for each image '''
    
    num_images = tf.shape[0]
    for i in range(num_images):
    
        print("Processing image: %d/%d" %(i+1,num_images))
        img = IO.readImage(inpath, scaling, idx=i)
            
        # Start and end of the exposure            
        t1 = tf[i]           
        t2 = t1 + te[i] + tr
        
        # Convert from seconds to samples
        n1 = int(t1/dt)
        n2 = int(t2/dt)
        ne = int(te[i]/dt)
        
        R = computeRotations(gyr[n1:n2+2,:],t[n1:n2+2])
        Bx, By = computeBlurfield(R,K,ne,height,width)

        IO.writeImage(img, outpath, 'blurred/', idx=i)
        IO.writeImage(Bx, outpath, 'blurx/', idx=i)
        IO.writeImage(By, outpath, 'blury/', idx=i)
        
        plotBlurVectors(Bx, By, img, outpath, idx=i) # Optional

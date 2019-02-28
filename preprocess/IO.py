import argparse
import numpy as np
import os
import glob
from PIL import Image

def parseInputs():
    desc = "Estimate gyro-based blur fields."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--input', type=str, metavar='', required=True, help='input folder')
    parser.add_argument('-o', '--output', type=str, metavar='', required=True, help='output folder')
    args = parser.parse_args()
    return args.input, args.output

def readGyroscope(inpath):
    datapath = inpath + '/imu/imu.txt'
    imu = np.loadtxt(datapath, dtype='float_', delimiter=' ')
    
    # The first column indicates the sensors type
    gyr_idx = imu[:,0].astype(np.int_) == 4
    
    # Extract gyroscope timestamps and readings
    tgyr = imu[gyr_idx,1]
    gyr = imu[gyr_idx,2:5]

    return gyr, tgyr
    
def readImageInfo(inpath):
    datapath = inpath + '/images/images.txt'
    info = np.loadtxt(datapath, dtype='float_', delimiter=' ')
    
    # Extract timestamps and exposure times
    tf = info[:,0]
    te = info[:,1]
    
    return tf, te

def readImage(inpath, scaling, idx):

    fnames = []
    for ext in ('*.jpg', '*.png'):
        datapath = inpath + '/images/' + ext
        fnames.extend(sorted(glob.glob(datapath)))
    
    if idx < len(fnames):
        img = Image.open(fnames[idx])
    else:
        raise ValueError('Could not read image with index: %d' %idx)
    
    # Downsample
    if scaling < 1.0:
        w = int(scaling*img.size[0])
        h = int(scaling*img.size[1])
        img = img.resize((w,h), resample=Image.BICUBIC)
        
    img = np.array(img)
    
    return img
    
    
def writeImage(img, outpath, folder, idx):
    
    fname = '%04d.png' %(idx)
    img = Image.fromarray(img.astype(np.uint8))
    img.save(outpath + '/' + folder + '/' + fname)
    
def createOutputFolders(outpath):

    try:
        os.makedirs(outpath + '/blurred')
        os.makedirs(outpath + '/blurx')
        os.makedirs(outpath + '/blury')
        os.makedirs(outpath + '/visualization/')
    except FileExistsError:
        # Directory already exists
        pass

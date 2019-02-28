import os, os.path, errno
import argparse
from PIL import Image
import numpy as np
from keras.models import Model
from keras.preprocessing.image import array_to_img
from models import modelsClass

# Parse input arguments
desc = "DeepGyro - Gyro-aided deblurring method."
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--input', type=str, metavar='', required=True, help='input folder')
args = parser.parse_args()
inpath = args.input

# Deblurred images will be saved to 'output' folder
outpath = "output"
try:
    os.makedirs(outpath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

blurred_path = inpath + "/blurred/"
blurx_path = inpath + "/blurx/"
blury_path = inpath + "/blury/"
              
blurred_names = os.listdir(blurred_path)
blurx_names = os.listdir(blurx_path)
blury_names = os.listdir(blury_path)
        
for fname in blurred_names:
        
    print("Deblurring '%s' with DeepGyro" %(fname))
                        
    blurred_img = Image.open(blurred_path + fname)
    blurx_img = Image.open(blurx_path + fname)
    blury_img = Image.open(blury_path + fname)
    
    blurred_np = (1./255)*np.array(blurred_img)
    blurx_np = (1./255)*np.array(blurx_img)
    blury_np = (1./255)*np.array(blury_img)
            
    width, height = blurred_img.size
    models = modelsClass(height,width)
    model = models.getDeepGyro()
    model.load_weights("checkpoints/DeepGyro.hdf5")
            
    b = np.reshape(blurred_np,[1,height,width,3])
    bx = np.reshape(blurx_np,[1,height,width,1])
    by = np.reshape(blury_np,[1,height,width,1]) 
    x = [b,bx,by]
    
    prediction = model.predict(x, batch_size=1,verbose=0,steps=None)
    prediction = prediction[0,:,:,:]
            
    deblurred_img = array_to_img(prediction)
    deblurred_img.save(outpath+"/%s"%(fname))
            
print("DONE!")

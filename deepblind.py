import os, os.path, errno
import argparse
from PIL import Image
import numpy as np
from keras.models import Model
from keras.preprocessing.image import array_to_img
from models import modelsClass

# Parse input arguments
desc = "DeepBlind - Blind deblurring method."
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
blurred_names = os.listdir(blurred_path)
        
for fname in blurred_names:
        
    print("Deblurring '%s' with DeepBlind" %(fname))
             
    blurred_img = Image.open(blurred_path + fname)
    blurred_np = (1./255)*np.array(blurred_img)
            
    width, height = blurred_img.size
    models = modelsClass(height,width)
    model = models.getDeepBlind()
    model.load_weights("checkpoints/DeepBlind.hdf5")
            
    x = np.reshape(blurred_np,[1,height,width,3])
    prediction = model.predict(x, batch_size=1,verbose=0,steps=None)
    prediction = prediction[0,:,:,:]
            
    deblurred_img = array_to_img(prediction)
    deblurred_img.save(outpath+"/%s"%(fname))
            
print("DONE!")

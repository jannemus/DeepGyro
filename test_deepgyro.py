import os, os.path, errno
from PIL import Image
import numpy as np
from keras.models import Model
from keras.preprocessing.image import array_to_img
from models import modelsClass

# Put your data to the input folder
inpath = "input"
outpath = "output/DeepGyro"

try:
    os.makedirs(outpath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

blurred_path = inpath + "/blurred/blurred/"
blurx_path = inpath + "/blurx/blurx/"
blury_path = inpath + "/blury/blury/"
              
blurred_names = os.listdir(blurred_path)
blurx_names = os.listdir(blurx_path)
blury_names = os.listdir(blury_path)
        
num_images = len(blurred_names)
print("Found %d inputs" %(num_images))
        
for i in range(0, num_images):
        
    print("Deblurring %s with DeepGyro" %(blurred_names[i]))
            
    path_blurred = blurred_path + blurred_names[i]
    path_blurx = blurx_path + blurx_names[i]
    path_blury = blury_path + blury_names[i]
            
    blurred_img = Image.open(path_blurred)
    blurx_img = Image.open(path_blurx)
    blury_img = Image.open(path_blury)
    
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
    deblurred_img.save(outpath+"/%s"%(blurred_names[i]))
            
print("DONE!")

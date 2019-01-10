import os, os.path, errno
from PIL import Image
import numpy as np
from keras.models import Model
from keras.preprocessing.image import array_to_img
from models import modelsClass

# Put your data to the input folder
inpath = "input"
outpath = "output/DeepBlind"

try:
    os.makedirs(outpath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

blurred_path = inpath + "/blurred/blurred/"
blurred_names = os.listdir(blurred_path)

num_images = len(blurred_names)
print("Found %d inputs" %(num_images))
        
for i in range(0, num_images):
        
    print("Deblurring %s with DeepBlind" %(blurred_names[i]))
            
    path_blurred = blurred_path + blurred_names[i]       
    blurred_img = Image.open(path_blurred)
    blurred_np = (1./255)*np.array(blurred_img)
            
    width, height = blurred_img.size
    models = modelsClass(height,width)
    model = models.getDeepBlind()
    model.load_weights("checkpoints/DeepBlind.hdf5")
            
    x = np.reshape(blurred_np,[1,height,width,3])
    prediction = model.predict(x, batch_size=1,verbose=0,steps=None)
    prediction = prediction[0,:,:,:]
            
    deblurred_img = array_to_img(prediction)
    deblurred_img.save(outpath+"/%s"%(blurred_names[i]))
            
print("DONE!")

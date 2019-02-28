import numpy as np
import matplotlib.pyplot as plt

# Visualize blur vectors and save the resulting image
# to the folder 'visualization'.
def plotBlurVectors(Bx, By, img, outpath, idx):

    Bx = Bx.astype(np.float_) - 128
    By = By.astype(np.float_) - 128
    
    # Display faded version of blurred image in the background
    img_bg = img.copy()
    img_bg = 0.5*img_bg.astype(np.float_) + 100
    img_bg = np.clip(img_bg,0,255).astype(np.uint8)
 
    # Specify pixels for which to plot the blur vectors
    xsteps = 7
    ysteps = 5
    h, w = img.shape[:2]
    xvec = np.linspace(w/(xsteps+1), w, xsteps, endpoint=False)
    yvec = np.linspace(h/(ysteps+1), h, ysteps, endpoint=False)
    xvec = xvec.astype(np.int_)
    yvec = yvec.astype(np.int_)
    X, Y = np.meshgrid(xvec,yvec)
    X = X.ravel()
    Y = Y.ravel()

    # Overlay blur vectors on the background image
    plt.figure(figsize=(10,10))
    plt.imshow(img_bg)
    for x,y in zip(X,Y):
        u = np.array([x, x+Bx[y,x]])
        v = np.array([y, y+By[y,x]])
        u = x + (u - np.mean(u)) # Centering
        v = y + (v - np.mean(v)) # Centering
        plt.plot(u,v,'r')
        
    plt.title('Image %d' %idx)
    plt.axis('off')
    
    # Save figure as image
    fname = '%04d.png' %(idx)    
    plt.savefig(outpath + '/visualization/' + fname, bbox_inches='tight')
    plt.close()
import numpy as np

# Rotates gyroscope measurements from the IMU frame 
# to the camera frame.
def alignSpatial(gyr, Ri):

    gyr = gyr.dot(Ri)

    return gyr


# Temporally aligns gyroscope and image timestamps. Gyroscope 
# measurements are upsampled. The sampling interval 'dt' is 
# the time difference between two consecutive row exposures.
def alignTemporal(gyr, tgyr, tf, te, tr, td, dt):

    # Convert from nanoseconds to seconds
    tgyr = 1e-9*(tgyr - tgyr[0])
    tf = 1e-9*(tf - tf[0])
    te = 1e-9*te

    tgyr = tgyr + td

    # The exposure of the first image starts
    t1 = tf[0]

    # The exposure of the last image ends
    t2 = tf[-1] + te[-1] + tr
    
    t = np.arange(t1,t2,dt)
    gyrx = np.interp(t,tgyr,gyr[:,0])
    gyry = np.interp(t,tgyr,gyr[:,1])
    gyrz = np.interp(t,tgyr,gyr[:,2])
    gyr_new = np.vstack((gyrx,gyry,gyrz)).T
    
    return gyr_new, t, tf, te


# Computes blurfield B = (Bx,By). The horizontal and vertical
# components of the blur are returned as grayscale images Bx and By.
def computeBlurfield(R,K,nexp,height,width):

    Kinv = np.linalg.inv(K)
    
    xi, yi = np.meshgrid(range(width),range(height))
    xi = xi.astype(np.float_)
    yi = yi.astype(np.float_)
    
    Bx = np.zeros((height,width),dtype=np.float_)
    By = np.zeros((height,width),dtype=np.float_)
    
    for row in range(height):
    
        x = xi[row,:]
        y = yi[row,:]
        z = np.ones(width, dtype=np.float_)
        X = np.vstack((x,y,z))
        
        R1 = R[:,:,row]
        R2 = R[:,:,row+nexp]
        H = K.dot(R2).dot(R1.T).dot(Kinv)
        Xp = H.dot(X)

        Xp[0,:] = Xp[0,:]/Xp[2,:]
        Xp[1,:] = Xp[1,:]/Xp[2,:]

        Bx[row,:] = np.around(Xp[0,:]-X[0,:])
        By[row,:] = np.around(Xp[1,:]-X[1,:])
        
    # Make sure that y-component is negative
    ypos = By > 0
    Bx[ypos] = -1*Bx[ypos]
    By[ypos] = -1*By[ypos]
    
    # Bx and By are saved as grayscale images. Cannot
    # have negative values so the value 128 is added.
    Bx = (Bx + 128).astype(np.uint8)
    By = (By + 128).astype(np.uint8)
    
    return Bx, By


 # Computes rotation of the camera during the image exposure 
 # by integrating gyroscope readings.
def computeRotations(gyr, t):

    N = gyr.shape[0]
    
    # Integrate gyroscope readings
    qts = np.zeros((N-1,4),dtype=np.float)
    q = np.array([1,0,0,0], dtype=np.float)
    dq_dt = np.zeros_like(q)
    
    for k in range(0,N-1):
    
        dt = t[k+1] - t[k]
        
        dq_dt[0] = -0.5*(q[1]*gyr[k,0]+q[2]*gyr[k,1]+q[3]*gyr[k,2])
        dq_dt[1] = 0.5*(q[0]*gyr[k,0]-q[3]*gyr[k,1]+q[2]*gyr[k,2])
        dq_dt[2] = 0.5*(q[3]*gyr[k,0]+q[0]*gyr[k,1]-q[1]*gyr[k,2])
        dq_dt[3] = -0.5*(q[2]*gyr[k,0]-q[1]*gyr[k,1]-q[0]*gyr[k,2])
        
        q = q + dq_dt*dt
        q = q / np.linalg.norm(q)
        qts[k,:] = q
        
    qts = qts.T

    # Quarternions to rotation matrices
    R = np.zeros((3,3,N-1), dtype=np.float)
    
    for k in range(0,N-1):
        R[0,0,k] = qts[0,k]**2 + qts[1,k]**2-qts[2,k]**2-qts[3,k]**2;
        R[0,1,k] = 2 * (qts[1,k] * qts[2,k] - qts[0,k] * qts[3,k]);
        R[0,2,k] = 2 * (qts[1,k] * qts[3,k] + qts[0,k] * qts[2,k]);
        R[1,0,k] = 2 * (qts[1,k] * qts[2,k] + qts[0,k] * qts[3,k]);
        R[1,1,k] = qts[0,k]**2 - qts[1,k]**2 + qts[2,k]**2 - qts[3,k]**2;
        R[1,2,k] = 2 * (qts[2,k] * qts[3,k] - qts[0,k] * qts[1,k]);
        R[2,0,k] = 2 * (qts[1,k] * qts[3,k] - qts[0,k] * qts[2,k]);
        R[2,1,k] = 2 * (qts[2,k] * qts[3,k] + qts[0,k] * qts[1,k]);
        R[2,2,k] = qts[0,k]**2 - qts[1,k]**2 - qts[2,k]**2 + qts[3,k]**2;
    
    return R
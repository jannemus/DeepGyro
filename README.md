# DeepGyro

Keras implementation of the deblurring method: <br>
*Gyroscope-Aided Motion Deblurring with Deep Networks* [[arXiv](https://arxiv.org/abs/1810.00986)]

## Testing

Download weights for DeepGyro from [here](https://www.dropbox.com/s/lwbi8r9dsa77btw/DeepGyro.hdf5?dl=0) and put them to the <font color='darkgreen'>*checkpoints*</font> folder. The weights for DeepBlind are also provided [here](https://www.dropbox.com/s/xzowi9k6syg8b6i/DeepBlind.hdf5?dl=0). <br> 

To deblur images in the folder <font color='darkgreen'>*input/paper*</font>, execute: <br>
`python deepgyro.py -i input/paper` <br> 

To deblur images with DeepBlind, execute: <br>
`python deepblind.py -i input/paper`

## Deblurring your own images

Before you can deblur your own images with DeepGyro, you need to compute gyro-based blur fields. You can use the code provided in the <font color='darkgreen'>*preprocess*</font> folder. Note that you will need gyroscope measurements, image timestamps, exposure times and some calibration information (see the next section). Once you have done this, put your data to the <font color='darkgreen'>*input/mydata*</font> folder and execute:

`python deepgyro.py -i input/mydata`

If you want to deblur images with DeepBlind, the blur fields are not needed. You can simply put your blurry images to <font color='darkgreen'>*input/mydata/blurred*</font> folder and execute:

`python deepblind.py -i input/mydata`

Make sure the folder structure is the same as in <font color='darkgreen'>*input/paper*</font>.

## Generation of blur fields

You can generate your own blur fields by using the code in the <font color='darkgreen'>*preprocess*</font> folder. The raw input data is expected to be in the same format as the sample data in the <font color='darkgreen'>*preprocess/myrawdata*</font> folder. Generate blur fields using the command:

`python generate.py -i myrawdata -o mydata`

The output folder <font color='darkgreen'>*mydata*</font> can be moved to <font color='darkgreen'>*input*</font> folder to be processed by DeepGyro. Before you run `generate.py` on your data make sure to provide all necessary information as described in the following:

**Blurry images** &ensp; *myrawdata/images/xxx.jpg* <br>
There can be more than one image. For example, you could capture a burst of images and deblur them all.

**Image timestamps and exposure times** &ensp; *myrawdata/images/images.txt* <br>
The text file <font color='darkgreen'>*images.txt*</font> should have the same number of lines as there are input images. Each line contains the timestamp and exposure time (in nanoseconds). Example line: 2705480529000 30000000.

**Inertial measurements** &ensp; *myrawdata/imu/imu.txt* <br>
The text file <font color='darkgreen'>*imu.txt*</font> contains inertial measurements including timestamps (in nanoseconds). Only gyroscope readings are required. The lines that start with “4” represent gyroscope. Example line: 4  358285205625  -1.5909724  -0.0620517  0.058423

**Calibration information** &ensp; *calibration.py* <br>
Update the calibration file <font color='darkgreen'>*calibration.py*</font> so that it corresponds to your setup. The file should include intrinsic camera parameters, camera readout time (rolling shutter skew), IMU-camera temporal offset and the rotation matrix that aligns the IMU frame with the camera frame.

## Citation

If you find our code helpful in your research or work, please cite our paper.

```
@InProceedings{Mustaniemi_2019_WACV,
  author = {Mustaniemi, Janne and Kannala, Juho and Särkkä, Simo and Matas, Jiri and Heikkilä, Janne},
  title = {Gyroscope-Aided Motion Deblurring with Deep Networks},
  booktitle = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  month = {January},
  year = {2019}
}
```

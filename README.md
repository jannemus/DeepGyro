# DeepGyro

Keras implementation of the deblurring method presented in:

Gyroscope-Aided Motion Deblurring with Deep Networks [[arXiv](https://arxiv.org/abs/1810.00986)]

## Testing

Download weights for DeepGyro [(link)](https://www.dropbox.com/s/lwbi8r9dsa77btw/DeepGyro.hdf5?dl=0) and put them to the `checkpoints` folder. The weights for DeepBlind are also provided [(link)](https://www.dropbox.com/s/xzowi9k6syg8b6i/DeepBlind.hdf5?dl=0).

Execute `test_deepgyro.py` or `test_deepblind.py` to deblur images.

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

# Deep Joint Demosaicking and Denoising

SiGGRAPH Asia 2016 - Conditionally Accepted

Michaël Gharbi <gharbi@mit.edu>
Gaurav Chaurasia
Sylvain Paris
Frédo Durand


### Installation and dependencies

This code uses the following external packages:
* Caffe, tested with the master branch, commit #42cd785
* Numpy
* Scikit-Image (for image I/O)

You can install the python packages via pip:
`pip install -r requirements.txt`


### Running

Use the `run.py` script, for example:

```shell
python run.py --input data/image1.png --output output --model models/noise_aware --noise 0.02
```

The noise parameter is given in the [0, 1] range

`--gpu`: By default the program will run on CPU, use this flag for GPU.


### Processing RGB image with ground-truth

If provided an RGB input, the program will assume it is a demosaicked image.
It will add noise if requested, mosaick-it, run our algorithm then compute PSNR.


### Processing RAW files

If provided a grayscale image, the program assumes it is a GRBG Bayer mosaic.
Use the `offset-x` and `offset-y` flag, if you need to shift the mosaick pattern.

Convert the RAW file to a grayscale Bayer image using DCRaw:

```shell
dcraw -T -d -W -6 {filename}
```

This input can then be fed to `run.py`

To produce a comparable output from DCRaw's demosaicking:

```shell
dcraw -T -o 0 -M -W -6  {filename}
```

### Output

The program outputs a horizontal stack of images with, in order:
ground-truth input, noise-corrupted input, corrupted mosacik,
denoised/demosaicked result, max-scaled difference map.


### Models

We provide two Caffe models in the `models/` directory. One has been trained
with no noise (`models/noise_agnostic`) the other (`noise_aware`) with noise
variances in the range [0, 20] (out of 255). The noise agnostic model will *not*
attempt to perform denoising.


### Known issues

* The noise-aware model is not our final model. In particular it does not learn
  the boundary conditions the way our noise-agnostic model does. This is why
  this implementation yields a cropped output.


### Issues?

Contact Michael Gharbi <gharbi@mit.edu>


***This code is outdated, check <https://github.com/mgharbi/demosaicnet> instead>***

---

# Deep Joint Demosaicking and Denoising

SiGGRAPH Asia 2016

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

Use the `bin/demosaick` script, for example:

```shell
python bin/demosaick --input data/test_images/000001.png --output output --model pretrained_models/bayer_noise --noise 0.02
```

For results on the Xtrans mosaick:

```shell
python bin/demosaick --input data/test_images/000001.png --output output --model pretrained_models/xtrans --mosaic_type xtrans
```

Run `python bin/demosaick -h` for details on the flags you can pass to this script, e.g.
`--gpu` will run the GPU version of the model.


### Processing RGB image with ground-truth

When provided with an RGB input, the program will assume it is a ground-truth demosaicked image.
It will add noise if requested, mosaick-it, run our algorithm then compute PSNR.

### Processing RAW files

When provided with a grayscale image, the program assumes it is a GRBG Bayer mosaic.
Use the `offset-x` and `offset-y` flag, if you need to shift the mosaick pattern.

```shell
python bin/demosaick --input data/test_raw_images/5d_mark_ii_iso800.tiff --output output --model pretrained_models/bayer --offset_x 1
```

To convert a RAW file to a grayscale Bayer image suitable as input, you can use DCRaw:

```shell
dcraw -T -d -W -6 {filename}
```

This input can then be fed to `bin/demosaick`

To produce a comparable output from DCRaw's demosaicking algorithm run:

```shell
dcraw -T -o 0 -M -W -6 {filename}
```

### Output

When the ground-truth is available, the program outputs a horizontal stack of
images with, in order: ground-truth input, noise-corrupted input, corrupted
mosaick, denoised/demosaicked result, max-scaled difference map.

If the ground-truth is not available, the program simply outputs a demosaicked image.


### Models

We provide three pre-trained Caffe models in the `pretrained_models/`
directory. `bayer` has been trained with no noise, and `bayer_noise`) with
noise variances in the range \[0, 20\] (out of 255). The noise agnostic model
will *not* attempt to perform denoising.


### Downloading the dataset.

You can download the full training dataset [here](https://groups.csail.mit.edu/graphics/demosaicnet/dataset.html).
Alternatively, run the download script provided with this code:

```shell
cd data
python download_dataset.py
```


### Training a new model

You will first need to generate a new network description by running 
```shell
python bin/create_net --output output/new_net 
```

The script is populated with sensible default, but check `python
bin/create_net -h` for details on the parameters you can alter.

Then, generate lmdb training and validation databases from the downloaded datasets:

```shell
python bin/convert_to_lmdb --input data/images/train --output data/db_train
python bin/convert_to_lmdb --input data/images/val --output data/db_val
```

You can now run the training code:

```shell
caffe train --solver output/new_net/solver.prototxt --log_dir output/new_net/log 
```

Here's a full example with the dummy database files provided:

```shell
python bin/create_net --output output/dummy_net --train_db data/dummy_train_db --test_db data/dummy_val_db
caffe train --solver output/dummy_net/solver.prototxt --log_dir output/dummy_net/log 
```

### Running into issues?

Contact Michael Gharbi <gharbi@mit.edu>


### Known issues:

* The latest version of CAFFE might break compatibility with this code, we tested against Caffe commit #42cd785
* If you run into python errors, not finding the demosaicnet packages, try: 

    export PYTHONPATH=$PYTHONPATH:""



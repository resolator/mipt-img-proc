# Task 3: image binarization
Implement one of popular image binarization methods.

## Usage

```
usage: otsu.py [-h] --img-path IMG_PATH [IMG_PATH ...] --save-to SAVE_TO [--bs BS]

OTSU binarization implementation with improvements.

optional arguments:
  -h, --help            show this help message and exit
  --img-path IMG_PATH [IMG_PATH ...]
                        Path to image to binarize.
  --save-to SAVE_TO     Path to save dir.
  --bs BS               Pass this arg to use blocks binarization. Defines block size.
                        Recommended: 45.

```

Without `--bs` arg it will use vanilla OTSU algorithm.

## Results

Resulted images are contained in `data` folder:
- `binarized` contains results for vanilla OTSU binarization.
- `binarized_modified` contains results for modified OTSU binarization.

#### Time estimation

To process all images from `data/source` the algorithm took:
- about `5.8632` seconds for vanilla OTSU (or `0.39088` seconds per image).
- about `26.9502` seconds for modified OTSU (or `1.79668` seconds per image).

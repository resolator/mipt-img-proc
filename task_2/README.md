# Task 2: fractal compression
Implement fractal compression for 256x256 grayscale images.

## Usage

```
usage: main.py [-h] --img-path IMG_PATH --save-to SAVE_TO [--block-size {4,8}] [--workers WORKERS]
               [--iters ITERS] [--shrink]

Script for fractal encode/decode.

optional arguments:
  -h, --help           show this help message and exit
  --img-path IMG_PATH  Path to input image.
  --save-to SAVE_TO    Path to save dir.
  --block-size {4,8}   Block size for patterns.
  --workers WORKERS    Number of threads to encode.
  --iters ITERS        Restoring iterations.
  --shrink             Shrink original image before work (speed up).
```

For debugging it's recommended to use `--shrink` option. 
To speed up processing set the `--workers` option according to number of available worker threads.

## Results

Resulted images are contained in `data` folder:
- `data/results/*/` - directories with resulted images for each iteration
- `data/source` - images for test

PSNR graphics for each image:

![psnr](data/results/psnr-joined.png)


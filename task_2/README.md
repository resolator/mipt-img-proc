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

Resulted images are contained in `data` folder.

Table with PSNR for each decoding iteration:

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 28.4984 | 29.1101 | 29.8427 | 30.5541 | 31.8447 | 33.2624 | 34.3425 | 34.8555 | 35.0469 | 35.0738 |


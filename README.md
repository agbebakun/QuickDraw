<!-- spell-checker:words quickdraw venv numpy tensorboard scikit-learn cairocffi DYLD -->

# Fork of QuickDraw

From: https://github.com/vietnh1009/QuickDraw

## Setup

Create the virtual environment and install dependencies:

```bash
# Setup
python3 -m venv .venv  # 3.13.7, project used 3.6
source .venv/bin/activate
pip install torch numpy opencv-python scikit-learn tensorboardX

# Additional dependencies from stroke_to_raster.py
brew install cairo
pip install rdp cairocffi
```


## Training

Training classes defined in `CLASSES` in [src/config.py](src/config.py):

```
["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer", "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]
```

Training data must be stored in the `data` folder as `data/full_numpy_bitmap_*.npy`, and can be downloaded from [https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap) and .

To re-run training:

```bash
#source .venv/bin/activate
python train.py
```

...which writes to `trained_models/whole_model_quickdraw` and also generates `trained_models/logs.txt`.

Note also that images for each class are present in `images/*.png` at 72x72 resolution, but these are only used by the `camera_app.py` example.


## Run the example painting app

```bash
#source .venv/bin/activate
python painting_app.py
```

If you receive an error about finding the cairo library, you can temporarily fix it by setting the `DYLD_FALLBACK_LIBRARY_PATH` environment variable to point to the *Homebrew* library path:

```bash
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
```

<!--
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH
-->


---

_(Original `README.md` follows)_

---

<p align="center">
 <h1 align="center">QuickDraw</h1>
</p>

[![GitHub stars](https://img.shields.io/github/stars/uvipen/QuickDraw)](https://github.com/uvipen/QuickDraw/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/uvipen/QuickDraw?color=orange)](https://github.com/uvipen/QuickDraw/network)
[![GitHub license](https://img.shields.io/github/license/uvipen/QuickDraw)](https://github.com/uvipen/QuickDraw/blob/master/LICENSE)

## Introduction

Here is my python source code for QuickDraw - an online game developed by google. with my code, you could: 
* **Run an app which you could draw in front of a camera (If you use laptop, your webcam will be used by default)**
* **Run an app which you could draw on a canvas**

## Camera app
In order to use this app, you need a pen (or any object) with blue, red or green color. When the pen (object) appears in front of camera, it will be caught and highlighted by an yellow circle. When you are ready for drawing, you need to press **space** button. When you want to stop drawing, press **space** again
Below is the demo by running the script **camera_app.py**:
<p align="center">
  <img src="demo/quickdraw.gif" width=600><br/>
  <i>Camera app demo</i>
</p>

## Drawing app
The script and demo will be released soon

## Dataset
The dataset used for training my model could be found at [Quick Draw dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn). Here I only picked up 20 files for 20 categories

## Categories:
The table below shows 20 categories my model used:

|           |           |           |           |
|-----------|:-----------:|:-----------:|:-----------:|
|   apple   |   book    |   bowtie  |   candle  |
|   cloud   |    cup    |   door    | envelope  |
|eyeglasses |  guitar   |   hammer  |    hat    |
| ice cream |   leaf    | scissors  |   star    |
|  t-shirt  |   pants   | lightning |    tree   |

## Trained models

You could find my trained model at **trained_models/whole_model_quickdraw**

## Training

You need to download npz files corresponding to 20 classes my model used and store them in folder **data**. If you want to train your model with different list of categories, you only need to change the constant **CLASSES** at **src/config.py** and download necessary npz files. Then you could simply run **python3 train.py**

## Experiments:

For each class, I take the first 10000 images, and then split them to training and test sets with ratio 8:2. The training/test loss/accuracy curves for the experiment are shown below:

<img src="demo/loss_accuracy_curves.png" width="800"> 

## Requirements

* **python 3.6**
* **cv2**
* **pytorch** 
* **numpy**

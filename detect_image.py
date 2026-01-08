# Classify image
# By Dan Jackson, 2026

import sys
import cv2
import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch

# Fix: pickle.UnpicklingError
from src import model
torch.serialization.add_safe_globals([model.QuickDraw, torch.nn.modules.container.Sequential, torch.nn.modules.conv.Conv2d, torch.nn.modules.activation.ReLU, torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.linear.Linear, torch.nn.modules.dropout.Dropout])

LINE_DIAMETER = 16
PADDING = 16
UNIT_SIZE = 256
CLASSIFY_SIZE = 28

CAPTURE_IMAGE_CROP_MARGIN = 0.25
CAPTURE_IMAGE_SHARPEN = True


# Classify
def classify(model, image):
    image = np.array(image, dtype=np.float32)[None, None, :, :]
    image = torch.from_numpy(image)
    logits = model(image)
    class_id = torch.argmax(logits[0])
    detected_class = CLASSES[class_id]
    return detected_class


# Skeletonize - from: https://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True
    return skel


# Normalize image for classification
def normalize_image(image, hack_sharpen = False, debugPrefix = None):
    w, h = image.shape[1], image.shape[0]

    # Greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)

    # Sharpen
    if hack_sharpen:
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.filter2D(image, -1, kernel)  # apply twice

    # Blur
    image = cv2.medianBlur(image, 5)  #image = cv2.GaussianBlur(image, (5,5), 0)
 
    # Dynamic binary thresholding
    #_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert image
    image = 255 - image
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".crop.png", image)

    # Skeletonize the image
    image = skeletonize(image)
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".skel.png", image)

    # Remove lone pixels
    kernel = np.array([ [-1, -1, -1],
                        [-1,  1, -1],
                        [-1, -1, -1] ], dtype="int")
    single_pixels = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    single_pixels_inv = cv2.bitwise_not(single_pixels)
    image = cv2.bitwise_and(image, image, mask=single_pixels_inv)

    # Find current bounding box
    min_y = 0
    max_y = h
    min_x = 0
    max_x = w
    ys, xs = np.nonzero(image)
    if len(xs) > 0:
        min_x = np.min(xs)
        max_x = np.max(xs)
    if len(ys) > 0:
        min_y = np.min(ys)
        max_y = np.max(ys)

    # Adjust so that bounding box has padding, is square, and centred
    box_width = max_x - min_x
    box_height = max_y - min_y
    box_size = max(box_width, box_height)
    if box_size < 1:
        box_size = 4
    box_size += (PADDING * 2 + LINE_DIAMETER) * box_size / UNIT_SIZE
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    min_x = int(center_x - box_size / 2)
    max_x = int(center_x + box_size / 2)
    min_y = int(center_y - box_size / 2)
    max_y = int(center_y + box_size / 2)

    # Crop image, with black where out of bounds
    cropped_image = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if 0 <= y < h and 0 <= x < w:
                cropped_image[y - min_y, x - min_x] = image[y, x]
    image = cropped_image
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".crop.png", image)

    # Dilate the image to the required line thickness
    dilate_radius = LINE_DIAMETER * box_size / UNIT_SIZE / 2
    kernel_size = int(round(dilate_radius * 2)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image = cv2.dilate(image, kernel)
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".dilate.png", image)

    # Resize
    image = cv2.resize(image, (CLASSIFY_SIZE, CLASSIFY_SIZE), interpolation=cv2.INTER_AREA)
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".norm.png", image)

    # Return
    return image


def crop_margin_proportion(image, margin_proportion):
    h, w = image.shape[0], image.shape[1]
    margin_x = int(w * margin_proportion)
    margin_y = int(h * margin_proportion)
    image = image[margin_y:h - margin_y, margin_x:w - margin_x]
    return image


def evaluate(filenames):
    # Load model
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage, weights_only=False)
    model.eval()

    hack_sharpen = False

    # Where no filenames were specified, capture using the camera
    capture_filename = "capture.png"
    if not filenames:
        print("CAMERA: Using camera to capture image...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return
        cv2.waitKey(1000)  # Wait for camera adjustments
        ret, image = cap.read()
        if not ret:
            print("ERROR: Could not read frame from camera.")
            return
        
        image = crop_margin_proportion(image, CAPTURE_IMAGE_CROP_MARGIN)

        cv2.imwrite(capture_filename, image)
        cap.release()
        filenames = [capture_filename]

    # Evaluate each image from the command line
    for filename in filenames:
        image = cv2.imread(filename)

        #debugPrefix = None
        debugPrefix = filename

        # HACK: Add a sharpen step to captured images
        if filename == capture_filename:
            hack_sharpen = CAPTURE_IMAGE_SHARPEN
        else:
            hack_sharpen = False
        
        normalized_image = normalize_image(image, hack_sharpen, debugPrefix)
        detected_class = classify(model, normalized_image)

        print(f"IMAGE: {filename} --> {detected_class}")


if __name__ == '__main__':
    filenames = sys.argv[1:]
    evaluate(filenames)

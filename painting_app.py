import cv2
import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch

from stroke_to_raster import stroke_to_raster

# Fix: pickle.UnpicklingError
from src import model
torch.serialization.add_safe_globals([model.QuickDraw, torch.nn.modules.container.Sequential, torch.nn.modules.conv.Conv2d, torch.nn.modules.activation.ReLU, torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.linear.Linear, torch.nn.modules.dropout.Dropout])


# Original resize method
def resize_image(image, size = 28):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(image)
    min_y = np.min(ys)
    max_y = np.max(ys)
    min_x = np.min(xs)
    max_x = np.max(xs)
    image = image[min_y:max_y, min_x: max_x]
    image = cv2.resize(image, (size, size))
    return image

# Classify
def classify(model, image):
    image = np.array(image, dtype=np.float32)[None, None, :, :]
    image = torch.from_numpy(image)
    logits = model(image)
    class_id = torch.argmax(logits[0])
    detected_class = CLASSES[class_id]
    return detected_class


def main():
    # Load model
    if torch.cuda.is_available():
        model = torch.load("trained_models/whole_model_quickdraw")
    else:
        model = torch.load("trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage, weights_only=False)
    model.eval()
    image = None
    cv2.namedWindow("Canvas")
    global ix, iy, is_drawing
    is_drawing = False
    image_changed = True
    # Track strokes
    strokes = []

    def paint_draw(event, x, y, flags, param):
        global ix, iy, is_drawing
        global image_changed
        if event == cv2.EVENT_LBUTTONDOWN:
            is_drawing = True
            ix, iy = x, y
            # Start new stroke
            strokes.append([])
            # Add point to current stroke
            strokes[-1].append([x, y])
            image_changed = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_drawing == True:
                cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                ix = x
                iy = y
                # Add point to current stroke
                strokes[-1].append([x, y])
                image_changed = True
        elif event == cv2.EVENT_LBUTTONUP:
            if is_drawing == True:
                # cv2.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                ix = x
                iy = y
                # Add point to current stroke
                strokes[-1].append([x, y])
                image_changed = True
            is_drawing = False
        return x, y

    cv2.setMouseCallback('Canvas', paint_draw)
    while (1):
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        # HACK: image_changed not being checked properly?
        image_changed = True
        if image_changed:
            cv2.imshow('Canvas', 255 - image)
            image_changed = False
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 127 or key == 8:
            image = None
            is_drawing = False
        elif key == ord(" "):
            # Original resize method
            image = resize_image(image)
            detected_class = classify(model, image)
            print("Original: " + detected_class)
            cv2.imshow('Canvas', image)
            cv2.waitKey(1000)

            # New resize method -- convert strokes to required format
            image = stroke_to_raster(strokes)
            detected_class = classify(model, image)
            print("New: " + detected_class)
            cv2.imshow('Canvas', image)
            cv2.waitKey(2000)

            # Clear canvas
            image = None
            is_drawing = False

            # Clear strokes
            strokes = []
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

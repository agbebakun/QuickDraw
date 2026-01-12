import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch

# Fix: pickle.UnpicklingError
from src import model
torch.serialization.add_safe_globals([model.QuickDraw, torch.nn.modules.container.Sequential, torch.nn.modules.conv.Conv2d, torch.nn.modules.activation.ReLU, torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.linear.Linear, torch.nn.modules.dropout.Dropout])


def load_model(filename):
    # Load model
    if torch.cuda.is_available():
        model = torch.load(filename)
    else:
        model = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=False)
    model.eval()
    return model


# Classify
def classify(model, image):
    image = np.array(image, dtype=np.float32)[None, None, :, :]
    image = torch.from_numpy(image)
    logits = model(image)
    # class_id = torch.argmax(logits[0])
    # detected_class = CLASSES[class_id]
    # return detected_class
    # Returning an ordered list with scores
    scores = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    class_scores = [(CLASSES[i], float(scores[i])) for i in range(len(CLASSES))]
    class_scores = sorted(class_scores, key=lambda x: x[1], reverse=True)
    return class_scores

# Most likely from classification scores
def most_likely(class_scores):
    return class_scores[0][0]

# Print scores
def print_scores(class_scores, threshold=0.001, top_k=5):
    print("Class scores:")
    count = 0
    for class_name, score in class_scores:
        if score >= threshold:
            print(f"  {class_name}: {score:.4f}")
            count += 1
            if count >= top_k:
                break

def classes():
    return CLASSES


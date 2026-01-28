import numpy as np
from src.config import *
from src.dataset import CLASSES
import torch

# Fix: pickle.UnpicklingError
from src import model
torch.serialization.add_safe_globals([model.QuickDraw, torch.nn.modules.container.Sequential, torch.nn.modules.conv.Conv2d, torch.nn.modules.activation.ReLU, torch.nn.modules.pooling.MaxPool2d, torch.nn.modules.linear.Linear, torch.nn.modules.dropout.Dropout])

DEFAULT_MODEL = "trained_models/whole_model_quickdraw"
IMAGE_SIZE = 28

def load_model(filename = DEFAULT_MODEL):
    # Load model
    if torch.cuda.is_available():
        model = torch.load(filename)
    else:
        model = torch.load(filename, map_location=lambda storage, loc: storage, weights_only=False)
    model.eval()
    return model


# Classify
def classify(model, image, alt_max = False):
    image = np.array(image, dtype=np.float32)[None, None, :, :]
    # Scale image to [0,1]
    image = image / 255.0
    image = torch.from_numpy(image)
    logits = model(image)

    # logit_scores = [(CLASSES[i], float(logits[0][i])) for i in range(len(CLASSES))]
    # print(logit_scores)

    # class_id = torch.argmax(logits[0])
    # detected_class = CLASSES[class_id]
    # return detected_class
    # Returning an ordered list with scores

    if alt_max:
        # Own implementation of Softmax
        logit_values = logits.detach().cpu().numpy()[0]
        logit_values = [float(v) for v in logit_values]
        max_logit = max(logit_values)
        #min_logit = min(logit_values)
        # Exp values
        exp_values = [pow(2.718281828459045, lv - max_logit) for lv in logit_values]
        sum_exp = sum(exp_values)
        scores = [ev / sum_exp for ev in exp_values]
        print(scores)
    else:
        scores = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        scores = [float(v) for v in scores]
  
    class_scores = [(CLASSES[i], float(scores[i])) for i in range(len(CLASSES))]
    class_scores = sorted(class_scores, key=lambda x: x[1], reverse=True)
    #print(class_scores)
    return class_scores

# Most likely from classification scores
def most_likely(class_scores):
    return class_scores[0][0]

# Print scores
def print_scores(class_scores, threshold=0.001, top_k=6):
    print("+----- Recognition -----+")
    count = 0
    for class_name, score in class_scores:
        if score >= threshold:
            percentage = int(round(score * 100, 0))
            if percentage > 0:
                print(f"|  {class_name:14s} {percentage:>3}%  |")
                count += 1
                if top_k and count >= top_k:
                    break
    print("+-----------------------+")

# Print pairwise scores
def print_pairwise_scores(class_scores, pairs):
    if pairs is None or len(pairs) == 0:
        return
    print("+------------ Pairwise Proportions -------------+")
    for (class1, class2) in pairs:
        score1 = 0.0
        score2 = 0.0
        for (cname, score) in class_scores:
            if cname == class1:
                score1 = score
            if cname == class2:
                score2 = score
        total = score1 + score2
        p1 = 0
        p2 = 0
        if total > 0:
            p1 = int(round(score1 / total * 100.0))
            p2 = int(round(score2 / total * 100.0))
        print(f"|  {class1:14s} {p1:>3.0f}%  |  {p2:>3.0f}% {class2:>14s}  |")
    print("+-----------------------+-----------------------+")
    

def classes():
    return CLASSES

import numpy as np
import tensorflow as tf


def get_predicted_breed(prediction, class_lookup):
    idx = np.argmax(prediction)
    return class_lookup[idx]

def get_top_5_predicted_breeds(prediction, class_lookup):
    idxs = np.argsort(prediction)[::-1][:5]
    top_5_breeds = []
    for idx in idxs:
        top_5_breeds.append((class_lookup[idx], prediction[idx]))
    return top_5_breeds
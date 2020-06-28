import argparse
import numpy as np
from utility import get_labels, process_image
from model import new_model, load_checkpoint, predict

# Define command arguments
parser = argparse.ArgumentParser()
parser.add_argument('image_path', type = str)
parser.add_argument('model_path', type = str)
parser.add_argument('--top_k', type = int, default=5)
parser.add_argument('--category_names', type = str, default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()


# Get command arguments
image_path = args.image_path
model_path = args.model_path
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

# Get classes and labels
category_mapping = get_labels(category_names)

# Predict image
probs, classes = predict(model_path, image_path, top_k, gpu)

# Associate probabilities to class names and print
labels = []
for i in classes:
    labels.append(category_mapping[i])

print(f"Top {top_k} most likely classes for {image_path}.")
for i in range(top_k):
    print(f"{i + 1} : {labels[i]} with probability {probs[i] * 100:.2f}%")
    
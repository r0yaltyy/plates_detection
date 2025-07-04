import os
from collections import Counter

dataset_dir = "/home/r0yaltyy/sobes/dataset/valid/labels"
class_counts = Counter()
for file in os.listdir(dataset_dir):
    if file.endswith(".txt"):
        with open(os.path.join(dataset_dir, file), "r") as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1
print("Распределение классов:", dict(class_counts))

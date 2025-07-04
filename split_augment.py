import os
import shutil
import random

# Пути
augmented_images_dir = "/home/r0yaltyy/sobes/dataset/augmented/images"
augmented_labels_dir = "/home/r0yaltyy/sobes/dataset/augmented/labels"
train_images_dir = "/home/r0yaltyy/sobes/dataset/train/images"
train_labels_dir = "/home/r0yaltyy/sobes/dataset/train/labels"
test_images_dir = "/home/r0yaltyy/sobes/dataset/test/images"
test_labels_dir = "/home/r0yaltyy/sobes/dataset/test/labels"
valid_images_dir = "/home/r0yaltyy/sobes/dataset/valid/images"
valid_labels_dir = "/home/r0yaltyy/sobes/dataset/valid/labels"

# Создание директорий, если они не существуют
for directory in [train_images_dir, train_labels_dir, test_images_dir, test_labels_dir, valid_images_dir, valid_labels_dir]:
    os.makedirs(directory, exist_ok=True)

# Список файлов
image_files = [f for f in os.listdir(augmented_images_dir) if f.endswith((".jpg", ".png"))]
random.shuffle(image_files)

# Распределение (80% train, 10% test, 10% valid)
total_files = len(image_files)
train_split = int(0.8 * total_files)
test_split = int(0.9 * total_files)

train_files = image_files[:train_split]
test_files = image_files[train_split:test_split]
valid_files = image_files[test_split:]

# Функция для копирования файлов
def copy_files(file_list, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
    for file in file_list:
        # Копирование изображения
        src_img_path = os.path.join(src_img_dir, file)
        dst_img_path = os.path.join(dst_img_dir, file)
        shutil.copy2(src_img_path, dst_img_path)

        # Копирование разметки
        label_file = file.replace(".jpg", ".txt").replace(".png", ".txt")
        src_label_path = os.path.join(src_label_dir, label_file)
        dst_label_path = os.path.join(dst_label_dir, label_file)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)

# Копирование файлов
copy_files(train_files, augmented_images_dir, augmented_labels_dir, train_images_dir, train_labels_dir)
copy_files(test_files, augmented_images_dir, augmented_labels_dir, test_images_dir, test_labels_dir)
copy_files(valid_files, augmented_images_dir, augmented_labels_dir, valid_images_dir, valid_labels_dir)

# Вывод статистики
print(f"Распределено файлов: {total_files}")
print(f"Train: {len(train_files)} изображений")
print(f"Test: {len(test_files)} изображений")
print(f"Valid: {len(valid_files)} изображений")

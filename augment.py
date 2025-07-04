import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Путь к исходным данным
input_dir = "/home/r0yaltyy/sobes/dataset/train/images"
output_dir = "/home/r0yaltyy/sobes/dataset/augmented/images"
labels_dir = "/home/r0yaltyy/sobes/dataset/train/labels"
aug_labels_dir = "/home/r0yaltyy/sobes/dataset/augmented/labels"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

# Определение аугментаций
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Переворот по горизонтали
    A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),  # Изменение яркости и контраста
    A.GaussNoise(p=0.3, var_limit=(10.0, 50.0)),  # Шум
    A.ShiftScaleRotate(p=0.3, shift_limit=0.1, scale_limit=0.1, rotate_limit=15),  # Сдвиг, масштабирование, поворот
    A.Blur(p=0.2, blur_limit=3),  # Лёгкое размытие
    A.CoarseDropout(p=0.2, max_holes=1, max_height=20, max_width=20),
    A.ColorJitter(p=0.3, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Изменение цвета
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Обработка каждого изображения и разметки
for img_file in os.listdir(input_dir):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        img_path = os.path.join(input_dir, img_file)
        label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Чтение изображения
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Чтение разметки
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    data = line.strip().split()
                    class_id = int(data[0])
                    bbox = [float(x) for x in data[1:5]]  # x_center, y_center, width, height
                    bboxes.append(bbox)
                    class_labels.append(class_id)

        # Применение аугментации
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_labels = augmented['class_labels']

        # Сохранение изображения
        aug_img_path = os.path.join(output_dir, f"aug_{img_file}")
        cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

        # Сохранение разметки
        aug_label_path = os.path.join(aug_labels_dir, f"aug_{img_file.replace('.jpg', '.txt').replace('.png', '.txt')}")
        with open(aug_label_path, "w") as f:
            for bbox, label in zip(aug_bboxes, aug_labels):
                f.write(f"{label} {' '.join(map(str, bbox))}\n")

print(f"Сгенерировано {len(os.listdir(output_dir))} аугментированных изображений.")

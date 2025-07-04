from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import cv2
import torch
import os
import numpy as np

# Загрузка моделей
model1 = YOLO("runs/detect/train7/weights/best.pt")
model2 = YOLO("runs/detect/train8/weights/best.pt")

# Путь к входному видео и выходному файлу
input_video_path = "/home/r0yaltyy/sobes/input/4.MOV"
output_dir = "/home/r0yaltyy/sobes/output/"
output_video_path = os.path.join(output_dir, "ensemble_4_optimized.mp4")

os.makedirs(output_dir, exist_ok=True)

# Открытие видеофайла
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Ошибка открытия видео!")
    exit()

# Получение параметров видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Инициализация видеопотока для записи
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Словарь цветов для всех классов (BGR формат)
class_colors = {
    "borsh": (0, 0, 255),          # Красный
    "chicken": (0, 255, 0),         # Зелёный
    "cup": (255, 165, 0),           # Оранжевый
    "cutlery": (255, 0, 0),         # Синий
    "empty cup": (255, 255, 0),     # Жёлтый
    "empty plate": (128, 0, 128),   # Фиолетовый
    "meat": (0, 255, 255),          # Голубой
    "salad": (255, 0, 255),         # Магента
    "salad balsamic": (128, 0, 0),  # Тёмно-красный
    "shot": (0, 128, 255),          # Светло-оранжевый
    "soup": (255, 128, 0),          # Тёмно-оранжевый
    "teapot": (0, 128, 128)         # Бирюзовый
}

# Обработка каждого кадра
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Обработка кадра {frame_count}...")

    # Предсказания от обеих моделей
    results1 = model1.predict(source=frame, conf=0.5)
    results2 = model2.predict(source=frame, conf=0.5)

    # Извлечение и фильтрация данных
    boxes1 = results1[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores1 = results1[0].boxes.conf.cpu().numpy()
    labels1 = results1[0].boxes.cls.cpu().numpy()
    mask1 = scores1 >= 0.5  # Фильтр по conf
    boxes1 = boxes1[mask1]
    scores1 = scores1[mask1]
    labels1 = labels1[mask1]

    boxes2 = results2[0].boxes.xyxy.cpu().numpy()
    scores2 = results2[0].boxes.conf.cpu().numpy()
    labels2 = results2[0].boxes.cls.cpu().numpy()
    mask2 = scores2 >= 0.5  # Фильтр по conf
    boxes2 = boxes2[mask2]
    scores2 = scores2[mask2]
    labels2 = labels2[mask2]

    # Нормализация координат для WBF
    img_width = frame_width
    img_height = frame_height
    boxes1_normalized = boxes1 / [img_width, img_height, img_width, img_height]
    boxes2_normalized = boxes2 / [img_width, img_height, img_width, img_height]

    # Проверка на пустые списки
    if len(boxes1) == 0 or len(boxes2) == 0:
        print(f"Предсказания пустые для кадра {frame_count}, пропускаем WBF.")
        continue

    # Применение WBF
    try:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            [boxes1_normalized, boxes2_normalized],
            [scores1, scores2],
            [labels1, labels2],
            weights=[0.4, 0.6],  # Веса для train1 и train4
            iou_thr=0.5,        # Порог пересечения
            skip_box_thr=0.5   # Минимальный confidence
        )

        # Фильтрация результатов WBF по skip_box_thr
        mask_fused = fused_scores >= 0.5
        fused_boxes = fused_boxes[mask_fused]
        fused_scores = fused_scores[mask_fused]
        fused_labels = fused_labels[mask_fused]

        # Денаormalization обратно в абсолютные координаты
        fused_boxes = fused_boxes * [img_width, img_height, img_width, img_height]

        # Визуализация результатов
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = map(int, box)
            class_name = results1[0].names[int(label)]
            color = class_colors.get(class_name, (255, 255, 255))  # Белый по умолчанию
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)  # Жирные подписи

    except Exception as e:
        print(f"Ошибка WBF для кадра {frame_count}: {e}")
        continue

    # Запись обработанного кадра
    out.write(frame)

# Освобождение ресурсов
cap.release()
out.release()
print(f"Видео сохранено как {output_video_path}")

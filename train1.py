from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model with enhanced augmentation and AdamW optimizer
results = model.train(
    data="/home/r0yaltyy/sobes/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="AdamW",  # Более эффективный оптимизатор для классификации
    lr0=0.001,         # Низкая начальная скорость обучения для стабильности
    lrf=0.01,          # Фактор финальной скорости обучения
    momentum=0.9,      # Увеличенный момент для лучшей сходимости
    weight_decay=0.0005,  # Регуляризация для предотвращения переобучения
    mosaic=0.7,  # Уменьшаем мозаику
    hsv_h=0.05,  # Снижаем до 5%
    flipud=0.05,
    fliplr=0.05,
    translate=0.05,
    scale=0.05,
    shear=0.0,
    perspective=0.0,
    mixup=0.0,        # Лёгкий mixup для улучшения классификации
    copy_paste=0.0     # Отключаем для избежания избыточности
)

# Run inference on a test image
results = model("/home/r0yaltyy/sobes/dataset/test/images/2_1_frame_21_0s_jpg.rf.01c2190ea94f39c5bb60a234910c8ac1.jpg")


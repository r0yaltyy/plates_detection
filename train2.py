from ultralytics import YOLO

# Load a COCO-pretrained YOLO11s model
model = YOLO("yolo11s.pt")

# Train the model with higher resolution and optimized parameters
results = model.train(
    data="/home/r0yaltyy/sobes/dataset/data.yaml",
    epochs=100,
    imgsz=960,         # Увеличенный размер изображения для лучшей локализации
    batch=8,           # Уменьшенный батч для компенсации большего imgsz
    optimizer="SGD",   # Оставляем SGD для стабильности с большим разрешением
    lr0=0.01,          # Стандартная скорость обучения
    lrf=0.1,           # Более высокий фактор финальной скорости
    momentum=0.937,    # Дефолтный момент
    weight_decay=0.001,  # Увеличенная регуляризация
    mosaic=0.5,        # Уменьшаем мозаику для больших изображений
    hsv_h=0.0,         # Отключаем, фокусируемся на локализации
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0.0,
    fliplr=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    mixup=0.0,
    copy_paste=0.0
)

# Run inference on a test image
results = model("/home/r0yaltyy/sobes/dataset/test/images/2_1_frame_21_0s_jpg.rf.01c2190ea94f39c5bb60a234910c8ac1.jpg")

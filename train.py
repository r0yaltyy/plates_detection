from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model with mosaic augmentation only
results = model.train(
    data="/home/r0yaltyy/sobes/dataset/data.yaml",
    epochs=300,
    imgsz=640,
    mosaic=1.0,  # Включаем мозаику с вероятностью 1.0
    hsv_h=0.0,   # Отключаем аугментацию по оттенку
    hsv_s=0.0,   # Отключаем аугментацию по насыщенности
    hsv_v=0.0,   # Отключаем аугментацию по яркости
    flipud=0.0,  # Отключаем вертикальный флип
    fliplr=0.0,  # Отключаем горизонтальный флип
    translate=0.0,  # Отключаем трансляции
    scale=0.0,   # Отключаем масштабирование
    shear=0.0,   # Отключаем сдвиг
    perspective=0.0,  # Отключаем перспективные искажения
    mixup=0.0,   # Отключаем mixup аугментацию
    copy_paste=0.0  # Отключаем copy-paste аугментацию
)

# Run inference
results = model("/home/r0yaltyy/sobes/dataset/test/images/2_1_frame_21_0s_jpg.rf.01c2190ea94f39c5bb60a234910c8ac1.jpg")

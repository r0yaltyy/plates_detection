from ultralytics import YOLO

# Load a COCO-pretrained YOLO11s model
model = YOLO("yolo11s.pt")

results = model.train(
    data="/home/r0yaltyy/sobes/dataset/data.yaml",
    epochs=150,
    imgsz=960,
    batch=8,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    auto_augment="randaugment",  # Автоматическая аугментация
    mixup=0.2,  # MixUp
    mosaic=0.8,  # Увеличенный Mosaic
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    flipud=0.0, fliplr=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0
)


# Run inference on a test image
results = model("/home/r0yaltyy/sobes/dataset/test/images/2_1_frame_21_0s_jpg.rf.01c2190ea94f39c5bb60a234910c8ac1.jpg")

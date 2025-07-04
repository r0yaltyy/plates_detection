from ultralytics import YOLO
import wandb

model = YOLO("runs/detect/train6/weights/best.pt")

results = model.train(
    data="/home/r0yaltyy/sobes/dataset/data.yaml",
    epochs=150,
    imgsz=960,
    batch=8,
    optimizer="SGD",
    lr0=0.001,  # Уменьшенная скорость
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    freeze=10,  # Заморозка первых 10 слоёв
    warmup_epochs=5
)

# Run inference on a test image
results = model("/home/r0yaltyy/sobes/dataset/test/images/2_1_frame_21_0s_jpg.rf.01c2190ea94f39c5bb60a234910c8ac1.jpg")

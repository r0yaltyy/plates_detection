from ultralytics import YOLO

# Load a model
model = YOLO("/home/r0yaltyy/sobes/runs/detect/train6/weights/best.pt")

# Predict with the model
results = model("/home/r0yaltyy/sobes/input/3_1.MOV", save=True, conf=0.5, device=0, vid_stride=3, imgsz=960
                )
